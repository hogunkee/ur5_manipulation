import numpy as np
from pcd_gen import *
from cpd import *

PCG = PointCloudGen()

for img_idx in range(100, 120):
    state_depth = np.load('state/%d.npy'%img_idx)
    pcd_s = PCG.pcd_from_depth(state_depth)
    np.save('data/pcd_s_%d.npy'%img_idx, np.array(pcd_s.points))
