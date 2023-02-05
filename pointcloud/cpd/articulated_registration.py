import open3d as o3d
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

class ArtRegistration():
    def __init__(self, source, targets, K, max_iterations=1000, tolerance=1e-5, gpu=True):
        xs = np.copy(targets.points)
        ys = np.copy(source.points)
        #xs = np.copy(targets)
        #ys = np.copy(source)
        gmm = GaussianMixture(n_components=K, covariance_type='full')
        gmm.fit(ys)
        pi = np.asarray(gmm.weights_)
        mu = np.asarray(gmm.means_)
        sigma = np.asarray(gmm.covariances_)

        M = ys.shape[0]
        Z = np.ones((M, K)) / K
        for k in range(K):
            Z[:, k] = pi[k] * multivariate_normal.pdf(ys, mean=mu[k], cov=sigma[k])
        Z = np.divide(Z, np.sum(Z, axis=1, keepdims=True))

        init_Z = np.copy(Z)
        Z = init_Z.astype(np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ys)   
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        adjW = np.zeros((ys.shape[0], ys.shape[0]))
        for j in range(ys.shape[0]):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[j], 20)
            adjW[j, idx[1:]] = 1
        alpha = 1.0
        W = alpha * adjW
        W = W.astype(np.float32)
        
        from .rigid_registration import RigidRegistration
        #if gpu:
        #    from .affine_part_registration_torch_cuda import AffinePartRegistration
        #else:
        #    from .affine_part_registration import AffinePartRegistration
            
        self.reg = RigidRegistration(K=K, **{'X': xs, 'Y': ys, 'Z': Z, 'adjW': W, 'max_iterations': max_iterations, 'tolerance': tolerance})
       
    def register(self, callback):
        TY, params = self.reg.register(callback)
        return TY, params


class ArtRegistrationColor():
    def __init__(self, source, targets, K, max_iterations=1000, tolerance=1e-5, gpu=True):
        xs_pos = np.copy(targets.points)
        xs_rgb = np.copy(targets.colors)
        ys_pos = np.copy(source.points)
        ys_rgb = np.copy(source.colors)
        xs = np.concatenate([xs_pos, xs_rgb], axis=1)
        ys = np.concatenate([ys_pos, ys_rgb], axis=1)

        gmm = GaussianMixture(n_components=K, covariance_type='full')
        gmm.fit(ys)
        pi = np.asarray(gmm.weights_)
        mu = np.asarray(gmm.means_)
        sigma = np.asarray(gmm.covariances_)

        M = ys.shape[0]
        Z = np.ones((M, K)) / K
        for k in range(K):
            Z[:, k] = pi[k] * multivariate_normal.pdf(ys, mean=mu[k], cov=sigma[k])
        Z = np.divide(Z, np.sum(Z, axis=1, keepdims=True))

        init_Z = np.copy(Z)
        Z = init_Z.astype(np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ys[:, :3])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        adjW = np.zeros((ys.shape[0], ys.shape[0]))
        for j in range(ys.shape[0]):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[j], 20)
            adjW[j, idx[1:]] = 1
        alpha = 1.0
        W = alpha * adjW
        W = W.astype(np.float32)
        
        from .rigid_registration_color import RigidRegistrationColor
        #if gpu:
        #    from .affine_part_registration_torch_cuda import AffinePartRegistration
        #else:
        #    from .affine_part_registration import AffinePartRegistration
            
        self.reg = RigidRegistrationColor(K=K, **{'X': xs, 'Y': ys, 'Z': Z, 'adjW': W, 'max_iterations': max_iterations, 'tolerance': tolerance})
       
    def register(self, callback):
        TY, params = self.reg.register(callback)
        return TY, params

class ArtRegistrationProjection():
    def __init__(self, source, targets, K, max_iterations=1000, tolerance=1e-5, gpu=True):
        xs_pos = np.copy(targets.points)
        xs_pos_proj = np.copy(xs_pos)
        xs_pos_proj[:, 2] = xs_pos[:, 2].mean() #0.273
        xs_rgb = np.copy(targets.colors)
        ys_pos = np.copy(source.points)
        ys_rgb = np.copy(source.colors)
        xs_raw = np.concatenate([xs_pos, xs_rgb], axis=1)
        xs_proj = np.concatenate([xs_pos_proj, xs_rgb], axis=1)
        ys = np.concatenate([ys_pos, ys_rgb], axis=1)

        gmm = GaussianMixture(n_components=K, covariance_type='full')
        gmm.fit(ys)
        pi = np.asarray(gmm.weights_)
        mu = np.asarray(gmm.means_)
        sigma = np.asarray(gmm.covariances_)

        M = ys.shape[0]
        Z = np.ones((M, K)) / K
        for k in range(K):
            Z[:, k] = pi[k] * multivariate_normal.pdf(ys, mean=mu[k], cov=sigma[k])
        Z = np.divide(Z, np.sum(Z, axis=1, keepdims=True))

        init_Z = np.copy(Z)
        Z = init_Z.astype(np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ys[:, :3])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        adjW = np.zeros((ys.shape[0], ys.shape[0]))
        for j in range(ys.shape[0]):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[j], 20)
            adjW[j, idx[1:]] = 1
        alpha = 1.0
        W = alpha * adjW
        W = W.astype(np.float32)
        
        from .rigid_registration_projection import RigidRegistrationProjection
        #if gpu:
        #    from .affine_part_registration_torch_cuda import AffinePartRegistration
        #else:
        #    from .affine_part_registration import AffinePartRegistration
            
        self.reg = RigidRegistrationProjection(K=K, **{'X_raw': xs_raw, 'X': xs_proj, 'Y': ys, 'Z': Z, 'adjW': W, 'max_iterations': max_iterations, 'tolerance': tolerance})
       
    def register(self, callback):
        TY, params = self.reg.register(callback)
        return TY, params
