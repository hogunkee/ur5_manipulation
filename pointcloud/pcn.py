import os
import sys
import argparse
import copy
import random

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as Data

sys.path.append('/home/gun/Desktop/PCN-PyTorch')
from models import PCN
from dataset import ShapeNet
from visualization import plot_pcd_one_view
from metrics.metric import l1_cd, l2_cd, emd, f_score

from matplotlib import pyplot as plt
from PIL import Image
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from pcd_gen import PointCloudGen


#CATEGORIES_PCN       = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel']
#CATEGORIES_PCN_NOVEL = ['bus', 'bed', 'bookshelf', 'bench', 'guitar', 'motorbike', 'skateboard', 'pistol']

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)


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
    pc = pc / (4*m)
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


def test_pc(pc, model, params, save=True):
    if save:
        image_dir = os.path.join(params.result_dir, 'image')
        output_dir = os.path.join(params.result_dir, 'output')
        make_dir(image_dir)
        make_dir(output_dir)

    with torch.no_grad():
        p = pc[np.newaxis, :].to(params.device)
        coarse_, fine_ = model(p)

        output_coarse = coarse_[0].detach().cpu().numpy()
        output_fine = fine_[0].detach().cpu().numpy()
        if save:
            plot_pcd_one_view(os.path.join(image_dir, '{:03d}.png'.format(params.index)), [pc, output_fine], ['Input', 'Output'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
            export_ply(os.path.join(output_dir, '{:03d}.ply'.format(prarams.index)), output_pc)
    return output_coarse, output_fine


def test(params, save=False):
    if save:
        make_dir(params.result_dir)

    print(params.exp_name)

    # load pretrained model
    model = PCN(16384, 1024, 4).to(params.device)
    model.load_state_dict(torch.load(params.ckpt_path))
    model.eval()

    PCG = PointCloudGen()
    gmm = GaussianMixture(n_components=3, covariance_type='full')

    goal_depth = np.load('goal/%d.npy'%params.index)
    state_depth = np.load('state/%d.npy'%params.index)
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
        #pcd_fillbottom = points_i
        #pcd_fillbottom = fill_bottom(points_i, args.n_bottom_points)
        #pcd_inp = pad_pcd(pc_norm(pcd_fillbottom), args.n_in_points)
        pcd_inp = pc_norm(pad_pcd(points_i, 2048))
        #pcd_inp = pc_norm(points_i)
        print('mean:', pcd_inp.mean(0))
        print('std:', pcd_inp.std(0))
        pcd_inp = torch.Tensor(pcd_inp)

        output_coarse, output_fine = test_pc(pcd_inp, model, params, save)
        visualize(pcd_inp)
        visualize(output_fine)
        visualize(output_coarse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion Testing')
    parser.add_argument('--index', type=int, default=100, help='Batch size for data loader')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--ckpt_path', type=str, default='/home/gun/Desktop/PCN-PyTorch/checkpoint/best_l1_cd.pth', help='The path of pretrained model.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=6, help='Num workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for testing')
    parser.add_argument('--save', type=bool, default=False, help='Saving test result')
    parser.add_argument('--novel', type=bool, default=False, help='unseen categories for testing')
    params = parser.parse_args()

    test(params, params.save)
