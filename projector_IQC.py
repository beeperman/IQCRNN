import cvxpy as cp
import numpy as np


class ProjectorIQC(object):
    eps = 1e-7
    def __init__(self, xi_dim, z_dim, env, Q10=None, Q20=None, Aphi=0, Bphi=1, loss=None):
        """
        Give projected results of
              ~        ~    ~         ~
        S =  {AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3, Q1, Q2}
        Non-tilded supoorted with wrapper (forward backward evaluation)

        Solve: (based on conservative LMI condition)
        min_{S, Q1, Q2}  sum_{V in S} ||V-V0||_F^2 + loss(Q1, Q2)
        s.t.             LMI(S, Q1, Q2) >= 0

        Inputs: (*required)
            *S, *AG, *BG, CG, Aphi, Bphi, Q10, Q20,
            loss(a function that takes cvxpy variables as input of form f(Q1, Q2) -> scalar)
            Q2_invert: treat Q2_inv as Q2.. Approximate lower right instead
        #################
        LMI:
        [
            [2Q1i - Q1i Q1 Q1i, 0, AT, CT],
            [0, 2Q2i - Q2i Q2 Q2i, BT, DT],
            [A,        B,          Q1, 0 ],
            [C,        D,          0,  Q2],
        ]

        A = [
            [AG + BG ~DK2 CG, BG ~CK1],
            [~BK2 CG,         ~AK    ]
        ]

        B = [
            [BG DK1 (Bphi - Aphi)/2],
            [BK1 (Bphi - Aphi)/2]
        ]

        C = [[DK3 CG,  CK2]]

        D = 0
        """

        self.xe_dim  = env.nxe
        self.xi_dim = xi_dim
        self.z_dim  = z_dim  # equals v_dim
        self.r_dim = env.nr
        self.psi_dim = env.npsi
        self.q_dim = env.nq

        self.Ae = env.Ae
        self.Be1 = env.Be1
        self.Be2 = env.Be2
        self.Ce1 = env.Ce1
        self.De1 = env.De1
        self.Ce2 = env.CG2 if env.CG2 is not None else np.block([[np.eye(self.x_dim), np.zeros(self.x_dim, self.psi_dim)]])
        self.M   = env.M

        self.Aphi = Aphi if np.size(Aphi) > 1 else Aphi * np.eye(self.z_dim)
        self.Bphi = Bphi if np.size(Bphi) > 1 else Bphi * np.eye(self.z_dim)

        self.AK  = None  #~
        self.BK1 = None
        self.BK2 = None  #~
        self.CK1 = None  #~
        self.DK1 = None
        self.DK2 = None  #~
        self.CK2 = None
        self.DK3 = None

        self.Q10 = Q10 if Q10 is not None else np.eye(self.xi_dim + self.xe_dim)
        self.Q20 = Q20 if Q20 is not None else np.eye(self.z_dim)

        self.loss = loss  # if None then no loss added..

        self.last_status = None

    def project(self, AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3, Q10=None, Q20=None, loss=None):
        """
              ~        ~    ~         ~
        S =  {AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3, Q1^, Q2^}, ^: not required

        Q10, Q20, loss will override default (Last Q1, Q2 and self.loss)
        """
        xi_dim = self.xi_dim
        xe_dim = self.xe_dim
        zeta_dim = xi_dim + xe_dim
        z_dim  = self.z_dim
        q_dim  = self.q_dim
        r_dim  = self.r_dim
        y_dim  = np.shape(self.Ce2)[0]
        u_dim  = np.shape(self.Be2)[1]

        Ae = self.Ae
        Be1 = self.Be1
        Be2 = self.Be2
        Ce1 = self.Ce1
        De1 = self.De1
        Ce2 = self.Ce2
        M = self.M

        Bphi = self.Bphi
        Aphi = self.Aphi

        Q1 = Q10 if Q10 is not None else self.Q10
        Q2 = Q20 if Q20 is not None else self.Q20
        Q1i = np.linalg.inv(Q1)
        Q2i = np.linalg.inv(Q2)
        loss = loss if loss is not None else self.loss

        ########## CVXPY ##########
        vAK  = cp.Variable((xi_dim, xi_dim))
        vBK1 = cp.Variable((xi_dim, z_dim))
        vBK2 = cp.Variable((xi_dim, y_dim))
        vCK1 = cp.Variable((u_dim, xi_dim))
        vDK1 = cp.Variable((u_dim, z_dim))
        vDK2 = cp.Variable((u_dim, y_dim))
        vCK2 = cp.Variable((z_dim, xi_dim))
        vDK3 = cp.Variable((z_dim, y_dim))

        vQ1  = cp.Variable((zeta_dim, zeta_dim), symmetric=True)
        vQ2  = cp.diag(cp.Variable((z_dim)))  ## TODO: verify

        lambda_IQC = cp.Variable() # the lagrange multiplier for including IQCs

        obj = cp.sum_squares(vAK - AK) + \
              cp.sum_squares(vBK1 - BK1) + \
              cp.sum_squares(vBK2 - BK2) + \
              cp.sum_squares(vCK1 - CK1) + \
              cp.sum_squares(vDK1 - DK1) + \
              cp.sum_squares(vDK2 - DK2) + \
              cp.sum_squares(vCK2 - CK2) + \
              cp.sum_squares(vDK3 - DK3) + \
              cp.sum_squares(vQ1 - Q1) + \
              cp.sum_squares(vQ2 - Q2)

        if loss:
            obj + loss(Q1, Q2)

        A = cp.bmat([
            [Ae + Be2 @ vDK2 @ Ce2, Be2 @ vCK1],
            [vBK2 @ Ce2,            vAK     ]
        ])

        B1 = np.block([[Be1],
                      [np.zeros((xi_dim, q_dim))]])

        B2 = cp.bmat([
            [Be2 @ vDK1 @ (Bphi - Aphi) / 2],
            [vBK1 @ (Bphi - Aphi) / 2]
        ])

        C1 = cp.bmat([[vDK3 @ Ce2, vCK2]])

        C2 = np.block([[Ce1, np.zeros((r_dim, xi_dim))]])

        D1 = np.zeros((z_dim, q_dim))

        D2 = np.zeros((z_dim, z_dim))

        D3 = De1

        D4 = np.zeros((r_dim, z_dim))

        dyn = cp.bmat([[A,  B1, B2],
                       [C1, D1, D2]])

        Q   = cp.bmat([[vQ1, np.zeros((zeta_dim, z_dim))],
                       [np.zeros((z_dim, zeta_dim)), vQ2]])

        R   = np.block([[np.eye(zeta_dim), np.zeros((zeta_dim, q_dim)), np.zeros((zeta_dim, z_dim))],
                        [np.zeros((z_dim, zeta_dim)), np.zeros((z_dim, q_dim)), np.eye(z_dim)],
                        [C2, D3, D4]])

        N   = cp.bmat([[2 * Q1i - Q1i @ vQ1 @ Q1i, np.zeros((zeta_dim, z_dim)), np.zeros((zeta_dim, r_dim))],
                       [np.zeros((z_dim, zeta_dim)), 2 * Q2i - Q2i @ vQ2 @ Q2i, np.zeros((z_dim, r_dim))],
                       [np.zeros((r_dim, zeta_dim)), np.zeros((r_dim, z_dim)),  -lambda_IQC*M]])
        # stability condition
        LMI = cp.bmat([
            [R.T @ N @ R, dyn.T],
            [dyn,         Q]
        ])

        cons = [LMI >> self.eps*np.eye(zeta_dim+q_dim+z_dim+zeta_dim+z_dim), vQ1 >> self.eps*np.eye(zeta_dim), cp.diag(vQ2) >= self.eps, lambda_IQC >= 0]
        prob = cp.Problem(cp.Minimize(obj), cons)
        result = prob.solve(solver="MOSEK")  # plz install MOSEK to your python https://www.cvxpy.org/install/index.html
        #result = prob.solve(solver="SCS")  # The SCS solver seems to be fast

        AK  = vAK .value
        BK1 = vBK1.value
        BK2 = vBK2.value
        CK1 = vCK1.value
        DK1 = vDK1.value
        DK2 = vDK2.value
        CK2 = vCK2.value
        DK3 = vDK3.value

        Q10 = vQ1.value
        Q20 = vQ2.value

        self.Q10 = Q10
        self.Q20 = Q20

        return [AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3]

    def updateRNN(self, rnn, sess):
        AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3 = rnn.get_weights(sess)[:-3]
        AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3 = self.project(AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3)
        rnn.set_weights(sess, AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3)




# Well maybe subclass more convenient :3
class NonTildeProjectorIQC(ProjectorIQC):
    def project(self, AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3, Q10=None, Q20=None, loss=None):
        """
              ~        ~    ~         ~
        S =  {AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3, Q1, Q2}
        """
        # forward conversion
        AK  = AK  + BK1 @ (self.Aphi + self.Bphi)/2 @ CK2
        BK2 = BK2 + BK1 @ (self.Aphi + self.Bphi)/2 @ DK3
        CK1 = CK1 + DK1 @ (self.Aphi + self.Bphi)/2 @ CK2
        DK2 = DK2 + DK1 @ (self.Aphi + self.Bphi)/2 @ DK3

        AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3 = super().project(AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3, Q10, Q20, loss)

        # backward conversion
        AK  = AK  - BK1 @ (self.Aphi + self.Bphi)/2 @ CK2
        BK2 = BK2 - BK1 @ (self.Aphi + self.Bphi)/2 @ DK3
        CK1 = CK1 - DK1 @ (self.Aphi + self.Bphi)/2 @ CK2
        DK2 = DK2 - DK1 @ (self.Aphi + self.Bphi)/2 @ DK3

        return [AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3]