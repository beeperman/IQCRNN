import cvxpy as cp
import numpy as np

import sys

import mosek.fusion as mf
from mosek.fusion import Expr as mfe


class Projector2(object):
    eps = 1e-5
    def __init__(self, xi_dim, z_dim, AG, BG, CG=None, Aphi=0, Bphi=1, Q10=None, Q20=None, loss=None, Q2_invert=False):
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
        self.xi_dim = xi_dim
        self.z_dim  = z_dim  # equals v_dim

        self.AG = AG
        self.BG = BG
        self.CG = CG if CG is not None else np.eye(self.x_dim)

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

        self.Q10 = Q10 if Q10 is not None else np.eye(self.xi_dim + self.x_dim)
        self.Q20 = Q20 if Q20 is not None else np.eye(self.z_dim)

        self.Q2_invert = Q2_invert

        self.loss = loss  # if None then no loss added..

        self.last_status = None

    def project(self, AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3, Q10=None, Q20=None, loss=None):
        """
              ~        ~    ~         ~
        S =  {AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3, Q1^, Q2^}, ^: not required

        Q10, Q20, loss will override default (Last Q1, Q2 and self.loss)
        """
        xi_dim = self.xi_dim
        x_dim  = self.x_dim
        z_dim  = self.z_dim
        y_dim  = np.shape(self.CG)[0]
        u_dim  = np.shape(self.BG)[1]

        AG = self.AG
        BG = self.BG
        CG = self.CG

        Bphi = self.Bphi
        Aphi = self.Aphi

        Q1 = Q10 if Q10 is not None else self.Q10
        Q2 = Q20 if Q20 is not None else self.Q20
        Q1i = np.linalg.inv(Q1)
        Q2i = np.linalg.inv(Q2)
        loss = loss if loss is not None else self.loss
        Q2_invert = self.Q2_invert

        ########## CVXPY ##########
        vAK  = cp.Variable((xi_dim, xi_dim))
        vBK1 = cp.Variable((xi_dim, z_dim))
        vBK2 = cp.Variable((xi_dim, y_dim))
        vCK1 = cp.Variable((u_dim, xi_dim))
        vDK1 = cp.Variable((u_dim, z_dim))
        vDK2 = cp.Variable((u_dim, y_dim))
        vCK2 = cp.Variable((z_dim, xi_dim))
        vDK3 = cp.Variable((z_dim, y_dim))

        vQ1  = cp.Variable((xi_dim+x_dim, xi_dim+x_dim), symmetric=True)
        vQ2  = cp.diag(cp.Variable((z_dim)))  ## TODO: verify

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
            [AG + BG @ vDK2 @ CG, BG @ vCK1],
            [vBK2 @ CG,            vAK     ]
        ])

        B = cp.bmat([
            [BG @ vDK1 @ (Bphi - Aphi) / 2],
            [vBK1 @ (Bphi - Aphi) / 2]
        ])

        C = cp.bmat([[vDK3 @ CG, vCK2]])

        D = np.zeros((z_dim, z_dim))

        # TODO: Make it Q11, Q22, Q33, Q44.
        if not Q2_invert:
            LMI = cp.bmat([
                [2 * Q1i - Q1i @ vQ1 @ Q1i , np.zeros((xi_dim+x_dim, z_dim)), A.T, C.T],
                [np.zeros((z_dim, xi_dim+x_dim)), 2 * Q2i - Q2i @ vQ2 @ Q2i,  B.T, D.T],
                [A, B, vQ1, np.zeros((xi_dim+x_dim, z_dim))],
                [C, D, np.zeros((z_dim, xi_dim+x_dim)), vQ2]
            ])
        else:
            LMI = cp.bmat([
                [2 * Q1i - Q1i @ vQ1 @ Q1i , np.zeros((xi_dim+x_dim, z_dim)), A.T, C.T],
                [np.zeros((z_dim, xi_dim+x_dim)), vQ2,  B.T, D.T],
                [A, B, vQ1, np.zeros((xi_dim+x_dim, z_dim))],
                [C, D, np.zeros((z_dim, xi_dim+x_dim)), 2 * Q2i - Q2i @ vQ2 @ Q2i]
            ])

        #cons = [LMI >> 0, vQ1 >> self.eps*np.eye(xi_dim+x_dim), cp.diag(vQ2) >= self.eps]
        cons = [LMI >> self.eps * np.eye(len(LMI)), vQ1 >> self.eps*np.eye(xi_dim+x_dim), cp.diag(vQ2) >= self.eps]
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

    def projectMOSEK(self, AK0, BK10, BK20, CK10, DK10, DK20, CK20, DK30, Q10=None, Q20=None, loss=None):
        # ignoring loss..
        xi_dim = self.xi_dim
        x_dim = self.x_dim
        z_dim = self.z_dim
        y_dim = np.shape(self.CG)[0]
        u_dim = np.shape(self.BG)[1]

        AG = self.AG
        BG = self.BG
        CG = self.CG

        Bphi = self.Bphi
        Aphi = self.Aphi

        Q1 = Q10 if Q10 is not None else self.Q10
        Q2 = Q20 if Q20 is not None else self.Q20
        Q1i = np.linalg.inv(Q1)
        Q2i = np.linalg.inv(Q2)
        loss = loss if loss is not None else self.loss
        Q2_invert = self.Q2_invert

        ###### MOSEK ######
        AK = mf.Matrix.dense(AK0 .astype(np.double))
        BK1= mf.Matrix.dense(BK10.astype(np.double))
        BK2= mf.Matrix.dense(BK20.astype(np.double))
        CK1= mf.Matrix.dense(CK10.astype(np.double))
        DK1= mf.Matrix.dense(DK10.astype(np.double))
        DK2= mf.Matrix.dense(DK20.astype(np.double))
        CK2= mf.Matrix.dense(CK20.astype(np.double))
        DK3= mf.Matrix.dense(DK30.astype(np.double))

        AG = mf.Matrix.dense(AG.astype(np.double))
        BG = mf.Matrix.dense(BG.astype(np.double))
        CG = mf.Matrix.dense(CG.astype(np.double))

        BphimAphid2 = mf.Matrix.dense((Bphi.astype(np.double) - Aphi.astype(np.double))/2)
        Bphi = mf.Matrix.dense(Bphi.astype(np.double))
        Aphi = mf.Matrix.dense(Aphi.astype(np.double))


        Q1 = mf.Matrix.dense(Q1 .astype(np.double))
        Q2d= np.diag(Q2).astype(np.double)

        # testing
        Q1it2 = mf.Matrix.dense(2*Q1i.astype(np.double))
        Q2it2 = mf.Matrix.dense(2*Q2i.astype(np.double))
        Q1i= mf.Matrix.dense(Q1i.astype(np.double))
        Q2i= mf.Matrix.dense(Q2i.astype(np.double))

        with mf.Model('projector2_msk') as M:
            # Construct the model.
            vAK = M.variable("AK", [xi_dim, xi_dim], mf.Domain.unbounded())
            vBK1= M.variable("BK1", [xi_dim, z_dim], mf.Domain.unbounded())
            vBK2= M.variable("BK2", [xi_dim, y_dim], mf.Domain.unbounded())
            vCK1= M.variable("CK1", [u_dim, xi_dim], mf.Domain.unbounded())
            vDK1= M.variable("DK1", [u_dim, z_dim], mf.Domain.unbounded())
            vDK2= M.variable("DK2", [u_dim, y_dim], mf.Domain.unbounded())
            vCK2= M.variable("CK2", [z_dim, xi_dim], mf.Domain.unbounded())
            vDK3= M.variable("DK3", [z_dim, y_dim], mf.Domain.unbounded())
            vQ1r= M.variable("Q1r", mf.Domain.inPSDCone(xi_dim + x_dim))
            vQ2d= M.variable("Q2d", z_dim, mf.Domain.greaterThan(self.eps))

            vQ1 = mfe.add(vQ1r, self.eps * np.eye(xi_dim + x_dim, dtype=np.double))
            vQ2 = self.msk_vec_to_diag_matrix(vQ2d)

            #vsqAK = M.variable("sqAK",  np.prod([xi_dim,xi_dim]),mf.Domain.unbounded())
            #vsqBK1= M.variable("sqBK1", np.prod([xi_dim, z_dim]),mf.Domain.unbounded())
            #vsqBK2= M.variable("sqBK2", np.prod([xi_dim, y_dim]),mf.Domain.unbounded())
            #vsqCK1= M.variable("sqCK1", np.prod([u_dim, xi_dim]),mf.Domain.unbounded())
            #vsqDK1= M.variable("sqDK1", np.prod([u_dim,  z_dim]),mf.Domain.unbounded())
            #vsqDK2= M.variable("sqDK2", np.prod([u_dim,  y_dim]),mf.Domain.unbounded())
            #vsqCK2= M.variable("sqCK2", np.prod([z_dim, xi_dim]),mf.Domain.unbounded())
            #vsqDK3= M.variable("sqDK3", np.prod([z_dim,  y_dim]),mf.Domain.unbounded())
            #vsqQ1 = M.variable("sqQ1",  (xi_dim + x_dim)**2     ,mf.Domain.unbounded())
            #vsqQ2d= M.variable("sqQ2d", z_dim                   ,mf.Domain.unbounded())

            #vsqsum = M.variable("sqsum", mf.Domain.unbounded())

            vsqsumAK = M.variable("sqsumAK",  mf.Domain.unbounded())
            vsqsumBK1= M.variable("sqsumBK1", mf.Domain.unbounded())
            vsqsumBK2= M.variable("sqsumBK2", mf.Domain.unbounded())
            vsqsumCK1= M.variable("sqsumCK1", mf.Domain.unbounded())
            vsqsumDK1= M.variable("sqsumDK1", mf.Domain.unbounded())
            vsqsumDK2= M.variable("sqsumDK2", mf.Domain.unbounded())
            vsqsumCK2= M.variable("sqsumCK2", mf.Domain.unbounded())
            vsqsumDK3= M.variable("sqsumDK3", mf.Domain.unbounded())
            vsqsumQ1 = M.variable("sqsumQ1",  mf.Domain.unbounded())
            vsqsumQ2d= M.variable("sqsumQ2d", mf.Domain.unbounded())

            #obj = mfe.sum(mfe.stack(0, [
            #    vsqAK ,
            #    vsqBK1,
            #    vsqBK2,
            #    vsqCK1,
            #    vsqDK1,
            #    vsqDK2,
            #    vsqCK2,
            #    vsqDK3,
            #    vsqQ1 ,
            #    vsqQ2d
            #]))

            #obj = vsqsum

            obj = mfe.sum(mfe.stack(0, [
                vsqsumAK ,
                vsqsumBK1,
                vsqsumBK2,
                vsqsumCK1,
                vsqsumDK1,
                vsqsumDK2,
                vsqsumCK2,
                vsqsumDK3,
                vsqsumQ1 ,
                vsqsumQ2d
            ]))

            M.objective("SumOfSqureDiff", mf.ObjectiveSense.Minimize, obj)

            # Construct ABCD matrices
            A = mfe.stack([
                [mfe.add(AG, mfe.mul(BG, mfe.mul(vDK2, CG))), mfe.mul(BG, vCK1)],
                [mfe.mul(vBK2, CG), vAK]
            ])
            AT = mfe.transpose(A)

            B = mfe.stack([
                [mfe.mul(BG, mfe.mul(vDK1, BphimAphid2))],
                [mfe.mul(vBK1, BphimAphid2)]
            ])
            BT = mfe.transpose(B)

            C = mfe.stack([[mfe.mul(vDK3, CG), vCK2]])
            CT = mfe.transpose(C)

            D = mfe.zeros([z_dim, z_dim])
            DT = mfe.transpose(D)

            if not Q2_invert:
                # current result generated with the following LMI def.
                #LMI = mfe.stack([
                #    [mfe.sub(Q1it2, mfe.mul(Q1i, mfe.mul(vQ1, Q1i))),
                #     mfe.zeros([xi_dim + x_dim, z_dim]), AT, CT],
                #    [mfe.zeros([z_dim, xi_dim + x_dim]), mfe.sub(Q2it2, mfe.mul(Q2i, mfe.mul(vQ2, Q2i))), BT, DT],
                #    [A, B, vQ1, mfe.zeros([xi_dim + x_dim, z_dim])],
                #    [C, D, mfe.zeros([z_dim, xi_dim + x_dim]), vQ2]
                #])
                rho = 1.0
                LMI = mfe.stack([
                    [mfe.mul(rho**2, mfe.sub(Q1it2, mfe.mul(Q1i, mfe.mul(vQ1, Q1i)))), mfe.zeros([xi_dim + x_dim, z_dim]), AT, CT],
                    [mfe.zeros([z_dim, xi_dim + x_dim]), mfe.sub(Q2it2, mfe.mul(Q2i, mfe.mul(vQ2, Q2i))), BT, DT],
                    [A, B, vQ1, mfe.zeros([xi_dim + x_dim, z_dim])],
                    [C, D, mfe.zeros([z_dim, xi_dim + x_dim]), vQ2]
                ])
            else:
                LMI = mfe.stack([
                    [mfe.sub(Q1it2, mfe.mul(Q1i, mfe.mul(vQ1, Q1i))), mfe.zeros([xi_dim + x_dim, z_dim]), AT, CT],
                    [mfe.zeros([z_dim, xi_dim + x_dim]), vQ2, BT, DT],
                    [A, B, vQ1, mfe.zeros([xi_dim + x_dim, z_dim])],
                    [C, D, mfe.zeros([z_dim, xi_dim + x_dim]), mfe.sub(Q2it2, mfe.mul(Q2i, mfe.mul(vQ2, Q2i)))]
                ])

            # Constraints
            M.constraint("conssqsumAK",  mfe.stack(0, 0.5, vsqsumAK , mfe.flatten(mfe.sub(vAK , AK ))), mf.Domain.inRotatedQCone(2+vAK .getSize()))
            M.constraint("conssqsumBK1", mfe.stack(0, 0.5, vsqsumBK1, mfe.flatten(mfe.sub(vBK1, BK1))), mf.Domain.inRotatedQCone(2+vBK1.getSize()))
            M.constraint("conssqsumBK2", mfe.stack(0, 0.5, vsqsumBK2, mfe.flatten(mfe.sub(vBK2, BK2))), mf.Domain.inRotatedQCone(2+vBK2.getSize()))
            M.constraint("conssqsumCK1", mfe.stack(0, 0.5, vsqsumCK1, mfe.flatten(mfe.sub(vCK1, CK1))), mf.Domain.inRotatedQCone(2+vCK1.getSize()))
            M.constraint("conssqsumDK1", mfe.stack(0, 0.5, vsqsumDK1, mfe.flatten(mfe.sub(vDK1, DK1))), mf.Domain.inRotatedQCone(2+vDK1.getSize()))
            M.constraint("conssqsumDK2", mfe.stack(0, 0.5, vsqsumDK2, mfe.flatten(mfe.sub(vDK2, DK2))), mf.Domain.inRotatedQCone(2+vDK2.getSize()))
            M.constraint("conssqsumCK2", mfe.stack(0, 0.5, vsqsumCK2, mfe.flatten(mfe.sub(vCK2, CK2))), mf.Domain.inRotatedQCone(2+vCK2.getSize()))
            M.constraint("conssqsumDK3", mfe.stack(0, 0.5, vsqsumDK3, mfe.flatten(mfe.sub(vDK3, DK3))), mf.Domain.inRotatedQCone(2+vDK3.getSize()))
            M.constraint("conssqsumQ1",  mfe.stack(0, 0.5, vsqsumQ1 , mfe.flatten(mfe.sub(vQ1 , Q1 ))), mf.Domain.inRotatedQCone(2+vQ1 .getSize()))
            M.constraint("conssqsumQ2d", mfe.stack(0, 0.5, vsqsumQ2d, mfe.flatten(mfe.sub(vQ2d, Q2d))), mf.Domain.inRotatedQCone(2+vQ2d.getSize()))

            #M.constraint("conssqsum",
            #             mfe.stack(0, [vsqsum,
            #                           mfe.flatten(mfe.sub(vAK , AK )),
            #                           mfe.flatten(mfe.sub(vBK1, BK1)),
            #                           mfe.flatten(mfe.sub(vBK2, BK2)),
            #                           mfe.flatten(mfe.sub(vCK1, CK1)),
            #                           mfe.flatten(mfe.sub(vDK1, DK1)),
            #                           mfe.flatten(mfe.sub(vDK2, DK2)),
            #                           mfe.flatten(mfe.sub(vCK2, CK2)),
            #                           mfe.flatten(mfe.sub(vDK3, DK3)),
            #                           mfe.flatten(mfe.sub(vQ1 , Q1 )),
            #                           mfe.flatten(mfe.sub(vQ2d, Q2d))]),
            #             mf.Domain.inQCone(1 +
            #                               vAK .getSize() +
            #                               vBK1.getSize() +
            #                               vBK2.getSize() +
            #                               vCK1.getSize() +
            #                               vDK1.getSize() +
            #                               vDK2.getSize() +
            #                               vCK2.getSize() +
            #                               vDK3.getSize() +
            #                               vQ1 .getSize() +
            #                               vQ2d.getSize()
            #                               ))


            #M.constraint("conssqAK",  mfe.stack(1, [mfe.mul(0.5, mfe.ones(vAK .getSize())), vsqAK , mfe.flatten(mfe.sub(vAK , AK ))]), mf.Domain.inRotatedQCone().axis(1))
            #M.constraint("conssqBK1", mfe.stack(1, [mfe.mul(0.5, mfe.ones(vBK1.getSize())), vsqBK1, mfe.flatten(mfe.sub(vBK1, BK1))]), mf.Domain.inRotatedQCone().axis(1))
            #M.constraint("conssqBK2", mfe.stack(1, [mfe.mul(0.5, mfe.ones(vBK2.getSize())), vsqBK2, mfe.flatten(mfe.sub(vBK2, BK2))]), mf.Domain.inRotatedQCone().axis(1))
            #M.constraint("conssqCK1", mfe.stack(1, [mfe.mul(0.5, mfe.ones(vCK1.getSize())), vsqCK1, mfe.flatten(mfe.sub(vCK1, CK1))]), mf.Domain.inRotatedQCone().axis(1))
            #M.constraint("conssqDK1", mfe.stack(1, [mfe.mul(0.5, mfe.ones(vDK1.getSize())), vsqDK1, mfe.flatten(mfe.sub(vDK1, DK1))]), mf.Domain.inRotatedQCone().axis(1))
            #M.constraint("conssqDK2", mfe.stack(1, [mfe.mul(0.5, mfe.ones(vDK2.getSize())), vsqDK2, mfe.flatten(mfe.sub(vDK2, DK2))]), mf.Domain.inRotatedQCone().axis(1))
            #M.constraint("conssqCK2", mfe.stack(1, [mfe.mul(0.5, mfe.ones(vCK2.getSize())), vsqCK2, mfe.flatten(mfe.sub(vCK2, CK2))]), mf.Domain.inRotatedQCone().axis(1))
            #M.constraint("conssqDK3", mfe.stack(1, [mfe.mul(0.5, mfe.ones(vDK3.getSize())), vsqDK3, mfe.flatten(mfe.sub(vDK3, DK3))]), mf.Domain.inRotatedQCone().axis(1))
            #M.constraint("conssqQ1",  mfe.stack(1, [mfe.mul(0.5, mfe.ones(vQ1 .getSize())), vsqQ1 , mfe.flatten(mfe.sub(vQ1 , Q1 ))]), mf.Domain.inRotatedQCone().axis(1))
            #M.constraint("conssqQ2d", mfe.stack(1, [mfe.mul(0.5, mfe.ones(vQ2d.getSize())), vsqQ2d, mfe.flatten(mfe.sub(vQ2d, Q2d))]), mf.Domain.inRotatedQCone().axis(1))

            #M.constraint("consLMI", LMI, mf.Domain.isTrilPSD())
            M.constraint("consLMI", mfe.sub(LMI, self.eps * np.eye(LMI.getShape()[0], dtype=np.double)), mf.Domain.isTrilPSD())
            #M.constraint("consQ1PSD", mfe.sub(vQ1, mfe.mul(self.eps, mf.Matrix.eye(xi_dim + x_dim))), mf.Domain.inPSDCone())

            self.verbose = True
            if self.verbose:
                M.setLogHandler(sys.stdout)
            #M.setSolverParam("presolveEliminatorMaxNumTries", 0)
            #M.setSolverParam("presolveLindepUse", "off")
            #M.setSolverParam("presolveUse", "off")

            # Accept approximate solutions
            AccSolutionStatus = M.getAcceptedSolutionStatus()
            M.acceptedSolutionStatus(AccSolutionStatus.Anything)

            M.solve()

            AK  = np.reshape(vAK .level(), [xi_dim,xi_dim])
            BK1 = np.reshape(vBK1.level(), [xi_dim, z_dim])
            BK2 = np.reshape(vBK2.level(), [xi_dim, y_dim])
            CK1 = np.reshape(vCK1.level(), [u_dim, xi_dim])
            DK1 = np.reshape(vDK1.level(), [u_dim,  z_dim])
            DK2 = np.reshape(vDK2.level(), [u_dim,  y_dim])
            CK2 = np.reshape(vCK2.level(), [z_dim, xi_dim])
            DK3 = np.reshape(vDK3.level(), [z_dim,  y_dim])
            Q10 = np.reshape(vQ1r.level(), [xi_dim+x_dim, xi_dim+x_dim]) + self.eps * np.eye(xi_dim+x_dim)
            Q20 = np.diag(vQ2d.level())

            self.Q10 = Q10
            self.Q20 = Q20

        return [AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3]

    def updateRNN(self, rnn, sess):
        AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3 = rnn.get_weights(sess)[:-3]
        #AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3 = self.project(AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3)
        AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3 = self.projectMOSEK(AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3)
        rnn.set_weights(sess, AK, BK1, BK2, CK1, DK1, DK2, CK2, DK3)

    def msk_vec_to_diag_matrix(self, vec):
        n = vec.getSize()

        i = np.diag(np.arange(n**2).reshape((n,n)))
        j = np.arange(n)

        Dn = mf.Matrix.sparse(i.tolist(), j.tolist(), 1.0)

        S = mfe.reshape(mfe.mul(Dn, vec), [n, n])

        return S



# Well maybe subclass more convenient :3
class NonTildeProjector2(Projector2):
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