import numpy as np
from pcd_gen import *
from cpd import *

PCG = PointCloudGen()

img_idx = 6
goal_depth = np.load('goal/%d.npy'%img_idx)
goal_rgb = np.array(Image.open('goal/%d.png'%img_idx))
state_depth = np.load('state/%d.npy'%img_idx)
state_rgb = np.array(Image.open('state/%d.png'%img_idx))

pcd_g = PCG.pcd_from_rgbd(goal_rgb, goal_depth)
pcd_s = PCG.pcd_from_rgbd(state_rgb, state_depth)
print(np.array(pcd_g.points).shape)
print(np.array(pcd_s.points).shape)
pcd_g = pcd_g.random_down_sample(sampling_ratio=0.3)
pcd_s = pcd_s.random_down_sample(sampling_ratio=0.3)
print(np.array(pcd_g.points).shape)
print(np.array(pcd_s.points).shape)
#pcd_g = PCG.pcd_from_pointset(pc_g)
#PCG.visualize(pcd_g)

K = 3
reg = ArtRegistrationColor(pcd_s, pcd_g, K, max_iterations=100, tolerance=1e-5, gpu=False)
sigma2_1 = reg.reg.sigma2_1
reg.register(visualize_color)
Z = reg.reg.Z#.numpy()
