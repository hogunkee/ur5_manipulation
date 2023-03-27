import argparse
import os
import sys
import copy
import random
import numpy as np
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append('/home/gun/Desktop/VE-PCN')
import shapenet_edge as shapenet_dataset
import model_edge as models
import util_edge as util
import ops

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from pcd_gen import PointCloudGen


parser = argparse.ArgumentParser()
parser.add_argument('--remove_point_num', type=int, default=512)
parser.add_argument('--cal_edge', action='store_true')
parser.add_argument('--test_unseen', action='store_true')
parser.add_argument('--train_seen', action='store_true') # action='store_true'
parser.add_argument('--loss_type', type=str,default='topnet')
parser.add_argument('--train_pcn', action='store_true')
parser.add_argument('--img_idx', type=int, default=100)
parser.add_argument('--n_in_points', type=int, default=2048)
parser.add_argument('--n_gt_points', type=int, default=2048)
parser.add_argument('--n_out_points', type=int, default=2048)
parser.add_argument('--n_bottom_points', type=int, default=512)
parser.add_argument('--eval_path', default='data/shapenetcore_partanno_segmentation_benchmark_v0_test2_edge_200_5.000000.h5')
# data/topnet_dataset2019/val_edge.h5
# data/shapenetcore_partanno_segmentation_benchmark_v0_test2_edge_200_5.000000.h5
parser.add_argument('--gpu', type=str,default='0')
parser.add_argument('--run_name', default='test')
parser.add_argument('--num_gpus', type=int, default=1)

parser.add_argument('--normalize_ratio', type=float,default=0.5)
#parser.add_argument('--pre_trained_model', default='model_ours_2048.pth.tar')
parser.add_argument('--pre_trained_model', default='model_pcn.pth.tar')
parser.add_argument('--grid_size', type=int,default=32)

parser.add_argument('--random_seed', type=int, default=42)

args = parser.parse_args()
if args.num_gpus==1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

chamfer_index=ops.chamferdist_index.chamferdist.ChamferDistance()

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

import open3d as o3d
def visualize(pointset):
    if pointset.shape[0]==3:
        pointset = pointset.transpose(1, 0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointset)
    o3d.visualization.draw_geometries([pcd])

def pc_norm(pc):
    """ pc: NxC, return NxC """
    #return (pc - pc.min()) / (pc.max() - pc.min()) - 1/2
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / (2*m)
    return pc

def fill_bottom(pcd, Nbottom):
    N = pcd.shape[0]
    pcd_proj = copy.deepcopy(pcd)
    pcd_proj[:, :2] = pcd[:, :2]
    z_noise = np.random.rand(N) * (pcd[:, 2].max() - pcd[:, 2].min()) * 0.05
    pcd_proj[:, 2] = pcd[:, 2].min() + z_noise

    print(pcd_proj.shape)
    range_max = (pcd[:, :2].max(0) - pcd[:, :2].min(0)) * 0.8 + pcd[:, :2].min(0)
    range_min = (pcd[:, :2].max(0) - pcd[:, :2].min(0)) * 0.2 + pcd[:, :2].min(0)
    pcd_proj = pcd_proj[(pcd_proj[:, :2] < range_max).all(1)]
    pcd_proj = pcd_proj[(pcd_proj[:, :2] > range_min).all(1)]
    print(pcd_proj.shape)

    choice = np.random.choice(pcd_proj.shape[0], Nbottom)
    pcd_bottom = pcd_proj[choice]
    return np.concatenate([pcd, pcd_bottom], 0)

def pad_pcd(pcd, Nin):
    N = pcd.shape[0]
    if N < Nin:
        ii = np.random.choice(N, Nin - N)
        choice = np.concatenate([range(N), ii])
        result = pcd[choice]
    else:
        choice = np.random.choice(N, Nin)
        result = pcd[choice]
    return result

def apply_net(net, args):
    PCG = PointCloudGen()
    gmm = GaussianMixture(n_components=3, covariance_type='full')

    img_idx = args.img_idx #105
    goal_depth = np.load('goal/%d.npy'%img_idx)
    state_depth = np.load('state/%d.npy'%img_idx)
    pcd_g = PCG.pcd_from_depth(goal_depth)
    pcd_s = PCG.pcd_from_depth(state_depth)
    print(np.array(pcd_g.points).shape)

    points = np.array(pcd_g.points)
    print(points.min(0))
    gmm.fit(points)
    pi = np.asarray(gmm.weights_)
    mu = np.asarray(gmm.means_)
    sigma = np.asarray(gmm.covariances_)

    M = points.shape[0]
    Z = np.ones((M, 3)) / 3
    for k in range(3):
        Z[:, k] = pi[k] * multivariate_normal.pdf(points, mean=mu[k], cov=sigma[k])
    Z = np.divide(Z, np.sum(Z, axis=1, keepdims=True))

    for i in range(3):
        points_i = points[Z.argmax(axis=1)==i]
        pcd_fillbottom = points_i #fill_bottom(points_i, args.n_bottom_points)
        pcd_inp = pad_pcd(pc_norm(pcd_fillbottom), args.n_in_points)

        inp = torch.Tensor([pcd_inp])
        inp = inp.cuda().transpose(2, 1).contiguous()
        pred, dens, dens_cls, reg, voxels, pred_edge, reg_edge, \
                       dens_cls_edge, dens_edge = net(inp, n_points=args.n_out_points)
        # pred: [32, 3, 2048]
        # dens: [32, 1, 32, 32, 32]
        # dens_cls: [32, 32, 32, 32]
        # reg: [32, 2048]
        # voxels: [32, 1, 32, 32, 32]
        # pred_edge: [32, 3, 2048]
        # reg_edge: [32, 2048]
        # dens_cls_edge: [32, 32, 32, 32]
        # dens_edge: [32, 1, 32, 32, 32]

        visualize(inp[0].cpu().numpy())
        visualize(pred[0].cpu().numpy())
        visualize(pred_edge[0].cpu().numpy())
    return pred, dens, dens_cls, reg, voxels, pred_edge, reg_edge, dens_cls_edge, dens_edge

def eval_net(net, args):
    net.eval()
    with torch.no_grad():
        return apply_net(net, args)

def test(args):
    net = models.GridAutoEncoderAdaIN(args,rnd_dim=2, adain_layer=3, ops=ops)
    if args.num_gpus > 1:
        net = torch.nn.DataParallel(net)  # ,device_ids=[0,1]
    net=net.cuda()
    net.apply(util.init_weights)

    if os.path.isfile(os.path.join('/home/gun/Desktop/VE-PCN/runs/{}/{}'.format(args.run_name,args.pre_trained_model))):
        checkpoint = torch.load(os.path.join('/home/gun/Desktop/VE-PCN/runs/{}/{}'.format(args.run_name,args.pre_trained_model)))
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.pre_trained_model))

    args.training = False
    _ = eval_net(net, args)

if __name__ == '__main__':
    test(args)
