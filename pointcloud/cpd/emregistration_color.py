import numpy as np
import numbers
from warnings import warn
from timeit import default_timer as timer   
import open3d as o3d
import torch   

def initialize_sigma2(X, Y):
    (N, D) = X.shape
    (M, _) = Y.shape
    diff_pos = X[None, :, :D-3] - Y[:, None, :D-3]
    err_pos = diff_pos ** 2

    diff_color = X[None, :, D-3:] - Y[:, None, D-3:]
    err_color = diff_color ** 2
    sigma2_1 = np.sum(err_pos) / (D * M * N)
    sigma2_2 = np.sum(err_color) / (3 * M * N)
    return sigma2_1, sigma2_2

class EMRegistrationColor(object):
    """
    Expectation maximization point cloud registration.

    Attributes
    ----------
    X: numpy array
        NxD array of target points.

    Y: numpy array
        MxD array of source points.

    Y: numpy array
        MxD array of transformed source points.

    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.

    N: int
        Number of target points.

    M: int
        Number of source points.

    D: int
        Dimensionality of source and target points

    iteration: int
        The current iteration throughout registration.

    max_iterations: int
        Registration will terminate once the algorithm has taken this
        many iterations.

    tolerance: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    q: float
        The objective function value that represents the misalignment between source
        and target point clouds.

    diff: float (positive)
        The absolute difference between the current and previous objective function values.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector of all 1s.

    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.

    Np: float (positive)
        The sum of all elements in P.

    """

    def __init__(self, X, Y, Z=None, adjW=None, K=1, sigma2=None, max_iterations=None, tolerance=None, w=None, *args, **kwargs):

        self.X = X
        self.Y = Y
        self.K = K
        
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        
        self.TY = np.zeros((self.K, self.M, self.D), dtype=np.float32)
        for k in range(self.K):
            self.TY[k, :, :] = np.copy(Y)

        self.D = 3
        self.Dc = 3
        self.sigma2_1, self.sigma2_2 = initialize_sigma2(X, Y)
        
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.P = np.zeros((self.M, self.N, self.K))
        self.Pt1 = np.zeros((self.N, self.K))
        self.P1 = np.zeros((self.M, self.K))
        self.Np = 0
        self.Z = np.ones((self.M, self.K)) / self.K if Z is None else Z
        self.adjW = np.zeros((self.M, self.M)) if adjW is None else adjW
        self.epsilon = 1.0

        self.vis = o3d.visualization.Visualizer()
        
    def register(self, callback=lambda **kwargs: None):
        self.vis.create_window()
        self.transform_point_cloud()
        """kwargs = {'iteration': self.iteration,
                  'error': self.q, 'X': self.X, 'Y': self.TY, 'Z': self.Z}
        callback(**kwargs)"""
            
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            #start = timer()
            self.iterate()
            if callable(callback):# and self.iteration  % 100 == 0:
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X, 'Y': self.Y, 'TY': self.TY, 'Z': self.Z,
                          'vis': self.vis}
                callback(**kwargs)
                #print(self.epsilon)
            #print("one iteration:", timer()-start)    
        
        """kwargs = {'iteration': self.iteration,
                  'error': self.q, 'X': self.X, 'Y': self.Y, 'Z': self.Z}
        callback(**kwargs)"""
        print("end iteration: ", self.iteration, self.diff, self.tolerance)
        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")   

    def update_variance(self):
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes.")

    def iterate(self):
        #start = timer()
        #print("expectation:", timer() - start)
        self.expectation()
        self.maximization()
        #self.weight_decaying()
        self.iteration += 1
        
    def weight_decaying(self):
        orgadjW = self.adjW / self.epsilon
        self.epsilon = max(self.epsilon * 0.99, 0.1)
        self.adjW = orgadjW * self.epsilon

    def expectation(self):
        P_1 = np.zeros((self.M, self.N, self.K))
        P_2 = np.zeros((self.M, self.N, self.K))
        for k in range(self.K):
            P_1[:, :, k] = np.sum((self.X[None, :, :self.D] 
                                   - self.TY[k][:, None, :self.D]) ** 2, axis=2)
            P_2[:, :, k] = np.sum((self.X[None, :, self.D: self.D + self.Dc] 
                                   - self.TY[k][:, None, self.D: self.D + self.Dc]) ** 2, axis=2)

        P = np.exp(-P_1 / (2 * self.sigma2_1))
        P = P * np.exp(-P_2 / (2 * self.sigma2_2))
        P = np.multiply(P, np.tile(self.Z[:, np.newaxis, :], (1, self.N, 1)))

        c = (2 * np.pi * self.sigma2_1) ** (self.D / 2)
        c = c * (2 * np.pi * self.sigma2_2) ** (self.Dc / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N

        den = np.sum(P, axis=(0, 2))
        den[den == 0] = np.finfo(float).eps
        den += c
        den = np.tile(den[np.newaxis, :, np.newaxis], (self.M, 1, self.K))
        
        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P)

        self.Xc = self.X[:, self.D: self.D + self.Dc]
        self.Yc = self.Y[:, self.D: self.D + self.Dc]
        
    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()
