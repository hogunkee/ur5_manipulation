from builtins import super
import numpy as np
from .emregistration import EMRegistration
import torch
#from .utility import

def transpose_matrix(w=[0.0, 0.0, 1.0], theta=0.0):
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * np.dot(W, W)
    return R

class RigidRegistration(EMRegistration):
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
        for k in range(self.K):
            Pk = self.P[..., k]
            Pk1 = self.P1[..., k]
            Npk = np.sum(Pk)
            
            muX = np.divide(np.sum(np.dot(Pk, self.X), axis=0), Npk)
            muY = np.divide(np.sum(np.dot(np.transpose(Pk), self.Y), axis=0), Npk)

            X_hat = self.X - np.tile(muX, (self.N, 1))
            Y_hat = self.Y - np.tile(muY, (self.M, 1))
            YPY = np.dot(np.transpose(Pk1), np.sum(
                np.multiply(Y_hat, Y_hat), axis=1))
            #self.X_hat.append(X_hat)
            #self.YPY.append(YPY)

            A = np.dot(np.transpose(X_hat), np.transpose(Pk))
            A = np.dot(A, Y_hat)
            #self.A.append(A)
            
            self.X_hat[k, :, :] = X_hat
            self.YPY[k] = YPY
            self.A[k, :, :] = A

            U, _, V = np.linalg.svd(A, full_matrices=True)
            C = np.ones((self.D, ))
            C[self.D-1] = np.linalg.det(np.dot(U, V))

            self.R[k] = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
            self.t[k] = np.transpose(muX) - np.dot(np.transpose(self.R[k]), np.transpose(muY))

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
                self.TY[k] = np.dot(self.Y, self.R[k]) + self.t[k]
            return
        else:
            return np.dot(Y, self.R) + self.t
        
    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the rigid transformation.
        See the update rule for sigma2 in Fig. 2 of of https://arxiv.org/pdf/0905.2635.pdf.
        """
        qprev = self.q
        
        q = 0
        sigma2 = 0
        for k in range(self.K):
            trAR = np.trace(np.dot(self.A[k], self.R[k]))
            xPx = np.dot(np.transpose(self.Pt1[..., k]), np.sum(
                np.multiply(self.X_hat[k], self.X_hat[k]), axis=1))
            q += xPx - 2 * trAR + self.YPY[k]
            sigma2 += xPx - trAR
        qz = -np.multiply(self.P1, np.log(self.Z + np.finfo(float).eps))
        qz = np.sum(qz)
        self.q = q / (2 * self.sigma2) + qz + self.D * self.Np/2 * np.log(self.sigma2)
        self.diff = np.abs(self.q - qprev)
        self.sigma2 = sigma2 / (self.Np * self.D)
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the affine transformation parameters.
        """
        return self.R, self.t
