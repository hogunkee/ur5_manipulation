import numpy as np
from pcd_gen import *
from cpd import *

PCG = PointCloudGen()

img_idx = 103
goal_depth = np.load('goal/%d.npy'%img_idx)
state_depth = np.load('state/%d.npy'%img_idx)

pcd_g = PCG.pcd_from_depth(goal_depth)
pcd_s = PCG.pcd_from_depth(state_depth)
print(np.array(pcd_g.points).shape)
print(np.array(pcd_s.points).shape)
pcd_g = pcd_g.random_down_sample(sampling_ratio=0.3)
pcd_s = pcd_s.random_down_sample(sampling_ratio=0.3)
print(np.array(pcd_g.points).shape)
print(np.array(pcd_s.points).shape)
#pcd_g = PCG.pcd_from_pointset(pc_g)
#PCG.visualize(pcd_g)

K = 3
reg = ArtRegistration(pcd_s, pcd_g, K, max_iterations=100, tolerance=1e-5, gpu=False)
sigma2 = reg.reg.sigma2
reg.register(visualize)
Z = reg.reg.Z.numpy()
