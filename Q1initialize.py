# Adopted from (Scherer et al.,1997)
import cvxpy as cp
import numpy as np


class Q1_init(object):
    eps = 1e-5
    def __init__(self, AG, BG, CG=None):
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

        self.x_dim  = np.shape(AG)[0]
        self.xi_dim = self.x_dim
        # self.xi_dim = states_size

        self.AG = AG
        self.BG = BG
        self.CG = CG if CG is not None else np.eye(self.x_dim)

    def compute(self):
        """
              ~        ~    ~         ~
        S =  {AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3, Q1^, Q2^}, ^: not required

        Q10, Q20, loss will override default (Last Q1, Q2 and self.loss)
        """
        xi_dim = self.xi_dim
        x_dim  = self.x_dim

        y_dim  = np.shape(self.CG)[0]
        u_dim  = np.shape(self.BG)[1]

        AG = self.AG
        BG = self.BG
        CG = self.CG

        ########## CVXPY ##########
        vX  = cp.Variable((x_dim, x_dim), symmetric=True)
        vY  = cp.Variable((x_dim, x_dim), symmetric=True)
        vK  = cp.Variable((x_dim, x_dim))
        vL  = cp.Variable((x_dim, y_dim))
        vM  = cp.Variable((u_dim, x_dim))
        vN  = cp.Variable((u_dim, y_dim))

        obj = 0

        # yPAy = y^T * P * A * y
        yPAy = cp.bmat([
            [AG @ vY + BG @ vM, AG + BG @ vN @ CG],
            [vK,                vX @ AG + vL @ CG]
        ])

        # yPy = y^T * P * y
        yPy = cp.bmat([
            [vY,            np.eye(x_dim)],
            [np.eye(x_dim), vX]
        ])

        # current result generated with following LMI1
        #LMI1 = cp.bmat([
        #    [yPy, yPAy.T],
        #    [yPAy, yPy]
        #])

        rho = 0.98
        LMI1 = cp.bmat([
                [rho ** 2 *yPy,  yPAy.T],
                [yPAy, yPy]
        ])

        LMI2 = yPy

        cons = [LMI1 >> 0, LMI2 >> 0]
        prob = cp.Problem(cp.Minimize(obj), cons)
        result = prob.solve(solver="MOSEK")  # plz install MOSEK to your python https://www.cvxpy.org/install/index.html
        #result = prob.solve(solver="SCS")  # The SCS solver seems to be fast
        # SVD based method to determine P
        # u, s, vh = np.linalg.svd(np.eye(x_dim) - vY.value @ vX.value, full_matrices=True)
        # V = np.block([[u @ np.diag(s), np.zeros((x_dim, xi_dim - x_dim))]])
        # U = np.block([[vh.T, np.zeros((x_dim, xi_dim - x_dim))]])
        # xhat = np.linalg.lstsq(vX.value @ V, U @ (V.T @ U - np.eye(xi_dim)))
        # P = np.block([[vX.value, U],
        #               [U.T, xhat]])
        # hand picked U and V
        U = vX.value
        V = np.linalg.inv(vX.value) - vY.value
        P = np.linalg.inv(np.block([[vY.value, V],[np.eye(x_dim), np.zeros((x_dim, x_dim))]])) @ \
            np.block([[np.eye(x_dim), np.zeros((x_dim, x_dim))],[vX.value, U]])
        Q1init = np.linalg.inv(P)
        return Q1init
