from builtins import super
import numpy as np
from .emregistration_projection import EMRegistrationProjection
import torch
#from .utility import

def transpose_matrix(w=[0.0, 0.0, 1.0], theta=0.0):
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * np.dot(W, W)
    return R

class RigidRegistrationProjection(EMRegistrationProjection):
    """
    Affine registration.

    Attributes
    ----------
    B: numpy array (semi-positive definite)
        DxD affine transformation matrix.

    t: numpy array
        1xD initial translation vector.

    YPY: float
        Denominator value used to update the scale factor.
        Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    X_hat: numpy array
        Centered target point cloud.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf

    """

    def __init__(self, R=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        """self.R = []
        self.t = []
        for k in range(self.K):
            #self.R.append(transpose_matrix(theta=2 * np.pi * np.random.rand()) if R is None else R)
            self.R.append(np.eye(self.D) if R is None else R)
            self.t.append(np.atleast_2d(np.zeros((1, self.D))) if t is None else t)"""
            
        self.R = np.zeros((self.K, self.D, self.D))
        self.t = np.zeros((self.K, 1, self.D))
        for k in range(self.K):
            self.R[k, :, :] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])    

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.
        """
        self.X_hat = np.zeros((self.K, self.N, self.D), dtype=np.float32)
        self.YPY = np.zeros((self.K), dtype=np.float32)
        self.A = np.zeros((self.K, self.D, self.D), dtype=np.float32)
        self.YcPYc = np.zeros((self.K), dtype=np.float32)
        self.Ac = np.zeros((self.K, self.Dc, self.Dc), dtype=np.float32)
        mat_proj = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])

        for k in range(self.K):
            Pk = self.P[..., k]
            Pk1 = self.P1[..., k]
            Npk = np.sum(Pk)
            
            muX = np.divide(np.sum(np.dot(Pk, self.X[:, :self.D]), axis=0), Npk)
            muY = np.divide(np.sum(np.dot(np.transpose(Pk), self.Y[:, :self.D]), axis=0), Npk)

            X_hat = self.X[:, :self.D] - np.tile(muX, (self.N, 1))
            Y_hat = self.Y[:, :self.D] - np.tile(muY, (self.M, 1))

            Y_hat_proj = np.dot(Y_hat, np.dot(self.R[k], mat_proj))
            YPY = np.dot(np.transpose(Pk1), np.sum(np.multiply(Y_hat_proj, Y_hat_proj), axis=1))
            YcPYc = np.dot(np.transpose(Pk1), np.sum(np.multiply(self.Yc, self.Yc), axis=1))

            A = np.dot(np.transpose(X_hat), np.transpose(Pk))
            A = np.dot(A, Y_hat)# Y_hat_proj)
            A = np.dot(mat_proj, A)
            Ac = np.dot(np.transpose(self.Xc), np.transpose(Pk))
            Ac = np.dot(Ac, self.Yc)
            
            self.X_hat[k, :, :] = X_hat
            self.YPY[k] = YPY
            self.A[k, :, :] = A

            self.YcPYc[k] = YcPYc
            self.Ac[k, :, :] = Ac

            U, _, V = np.linalg.svd(A, full_matrices=True)
            C = np.ones((self.D, ))
            C[self.D-1] = np.linalg.det(np.dot(U, V))

            self.R[k] = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
            self.t[k] = np.transpose(muX) - np.dot(np.transpose(self.R[k]), np.transpose(muY))
            #self.t[k][:, 2] = 0.3

        Z = np.sum(self.P, axis=1) # N sum 
        den = np.sum(Z, axis=1, keepdims=True) #  NK sum
        regul_Z = np.dot(self.adjW, self.Z)
        regul_den = np.sum(regul_Z, axis=1, keepdims=True)
        self.Z = np.divide(Z + regul_Z, den + regul_den)

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the affine transformation.
        """
        if Y is None:
            for k in range(self.K):
                self.TY[k][:, :self.D] = np.dot(self.Y[:, :self.D], self.R[k]) + self.t[k]
            return
        else:
            return np.dot(Y[:, :self.D], self.R) + self.t
        
    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the rigid transformation.
        See the update rule for sigma2 in Fig. 2 of of https://arxiv.org/pdf/0905.2635.pdf.
        """
        qprev = self.q
        
        q_1 = 0
        q_2 = 0
        sigma2_1 = 0
        sigma2_2 = 0
        for k in range(self.K):
            trAR = np.trace(np.dot(self.A[k], self.R[k]))
            trAc = np.trace(self.Ac[k])
            xPx = np.dot(np.transpose(self.Pt1[..., k]), np.sum(
                np.multiply(self.X_hat[k][:, :2], self.X_hat[k][:, :2]), axis=1))
            xcPxc = np.dot(np.transpose(self.Pt1[..., k]), np.sum(
                np.multiply(self.Xc, self.Xc), axis=1))

            q_1 += xPx - 2 * trAR + self.YPY[k]
            q_2 += xcPxc - 2 * trAc + self.YcPYc[k]
            sigma2_1 += xPx - trAR
            sigma2_2 += xcPxc - trAc

        qz = -np.multiply(self.P1, np.log(self.Z + np.finfo(float).eps))
        qz = np.sum(qz)

        q_1 = q_1 / (2 * self.sigma2_1) + qz + self.D * self.Np/2 * np.log(self.sigma2_1)
        q_2 = q_2 / (2 * self.sigma2_2) + self.Dc * self.Np/2 * np.log(self.sigma2_2)
        self.q = q_1 + q_2
        self.diff = np.abs(self.q - qprev)

        self.sigma2_1 = sigma2_1 / (self.Np * self.D)
        self.sigma2_2 = sigma2_2 / (self.Np * 3)
        if self.sigma2_1 <= 0:
            self.sigma2_1 = self.tolerance / 10
        if self.sigma2_2 <= 0:
            self.sigma2_2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the affine transformation parameters.
        """
        return self.R, self.t
