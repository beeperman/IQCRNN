# Adopted from (Scherer et al.,1997)
import cvxpy as cp
import numpy as np


class Q1_init(object):
    eps = 1e-5
    def __init__(self, env):
        """
        Give projected results of
              ~        ~    ~         ~
        S =  {AK, BK1,s BK2, CK1, DK1, DK2, CK2, DK3, Q1, Q2}
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
        self.xi_dim = self.xe_dim
        self.q_dim = env.nq
        self.r_dim = env.nr

        self.Ae = env.Ae
        self.Be1 = env.Be1
        self.Be2 = env.Be2 / env.factor
        self.Ce1 = env.Ce1
        self.De1 = env.De1
        self.Ce2 = env.Ce2 if env.CG2 is not None else np.block([[np.eye(self.x_dim), np.zeros(self.x_dim, self.psi_dim)]])
        self.MDelta   = env.M

    def compute(self):
        """
              ~        ~    ~         ~
        S =  {AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3, Q1^, Q2^}, ^: not required

        Q10, Q20, loss will override default (Last Q1, Q2 and self.loss)
        """
        xi_dim = self.xi_dim
        xe_dim  = self.xe_dim
        zeta_dim = xe_dim + xi_dim
        q_dim = self.q_dim
        r_dim  = self.r_dim

        y_dim  = np.shape(self.Ce2)[0]
        u_dim  = np.shape(self.Be2)[1]

        Ae = self.Ae
        Be1 = self.Be1
        Be2 = self.Be2
        Ce1 = self.Ce1
        De1 = self.De1
        Ce2 = self.Ce2
        D = De1
        K = np.array([[-1.5]]) # static feedback gain
        MDelta   = self.MDelta

        ########## CVXPY ##########
        vP  = cp.Variable((xe_dim, xe_dim), symmetric=True)
        lambda_IQC = cp.Variable() # the lagrange multiplier for including IQCs
        M = lambda_IQC * MDelta

        obj = 0
        A = Ae + Be2 @ K @ Ce2
        B = Be1
        C = Ce1
        D = De1

        rho = 1.0
        LMI = cp.bmat([[A.T @ vP @ A - rho**2 * vP, A.T @ vP @ B],
                       [B.T @ vP @ A, B.T@ vP @ B]]) + \
              cp.bmat([[C.T @ M @ C, C.T @ M @ D],
                       [D.T @ M @ C, D.T @ M @ D]])

        cons = [LMI << 0, vP >> 1e-5*np.eye(xe_dim), lambda_IQC >= 0]
        prob = cp.Problem(cp.Minimize(obj), cons)
        result = prob.solve(solver="MOSEK")  # plz install MOSEK to your python https://www.cvxpy.org/install/index.html
        print(vP.value)
        P = vP.value
        Q1init = np.linalg.inv(P)
        return Q1init
