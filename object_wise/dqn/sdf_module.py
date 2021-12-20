import os
import sys
import cv2
import torch
import skfmm
import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

file_path = os.path.dirname(os.path.abspath(__file__))
# file_path = 'home/gun/Desktop/ur5_manipulation/object_wise/dqn'
sys.path.append(os.path.join(file_path, '../..', 'UnseenObjectClustering'))
import networks
from fcn.test_dataset import clustering_features #, test_sample
from fcn.config import cfg_from_file

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class SDFModule():
    def __init__(self):
        self.pretrained = os.path.join(file_path, '../..', 'UnseenObjectClustering', \
                'experiments/checkpoints/seg_resnet34_8s_embedding_cosine_color_sampling_epoch_16.checkpoint.pth')
        self.cfg_file = os.path.join(file_path, '../..', 'UnseenObjectClustering', \
                'experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml')
        cfg_from_file(self.cfg_file)

        self.network_name = 'seg_resnet34_8s_embedding'
        self.network_data = torch.load(self.pretrained)
        self.network = networks.__dict__[self.network_name](2, 64, self.network_data).type(dtype)
        self.network.eval()

        self.network_crop = None
        self.target_resolution = 96

    def get_masks(self, image, data_format='HWC'):
        if data_format=='HWC':
            image = image.transpose([2, 0, 1])
        im_tensor = torch.from_numpy(image).type(dtype).unsqueeze(0)
        features = self.network(im_tensor, None).detach()
        out_label, selected_pixels = clustering_features(features, num_seeds=100)

        segmap = out_label.cpu().detach().numpy()[0]
        num_blocks = int(segmap.max())
        masks = []
        for nb in range(1, num_blocks+1):
            _mask = (segmap == nb).astype(float)
            masks.append(_mask)
        features = features.cpu().numpy()[0]
        return masks, features

    def get_sdf(self, masks):
        sdfs = []
        for seg in masks:
            sd = skfmm.distance(seg.astype(int) - 0.5, dx=1)
            sdfs.append(sd)
        return np.array(sdfs) 

    def get_sdf_features(self, image, data_format='HWC', resize=True):
        if data_format=='HWC':
            image[:20] = [0.81960784, 0.93333333, 1.]
            image = image.transpose([2, 0, 1])
        masks, features = self.get_masks(image, data_format='CHW')
        sdfs = self.get_sdf(masks)

        rgb_features = []
        block_features = []
        for sdf in sdfs:
            local_rgb = image[:, sdf>=0].mean(1)
            local_feature = features[:, sdf>=0].mean(1)
            rgb_features.append(local_rgb)
            block_features.append(local_feature)
        rgb_features = np.array(rgb_features)
        block_features = np.array(block_features)

        if resize:
            res = self.target_resolution
            sdfs_resized = []
            for sdf in sdfs:
                resized = cv2.resize(sdf, (res, res), interpolation=cv2.INTER_AREA)
                sdfs_resized.append(resized)
            sdfs_raw = sdfs
            sdfs = np.array(sdfs_resized)/400.

        return sdfs, sdfs_raw, (rgb_features, block_features)
    
    def object_matching(self, features_src, features_dest):
        rgb_src, cnn_src = features_src
        rgb_dest, cnn_dest = features_dest
        concat_src = np.concatenate([rgb_src, cnn_src], axis=1)
        concat_dest = np.concatenate([rgb_dest, cnn_dest], axis=1)
        src_norm = concat_src / np.linalg.norm(concat_src, axis=1).reshape(len(rgb_src), 1)
        dest_norm = concat_dest / np.linalg.norm(concat_dest, axis=1).reshape(len(rgb_dest), 1)
        #idx_src2dest = src_norm.dot(dest_norm.T).argmax(0)
        #idx_dest2src = src_norm.dot(dest_norm.T).argmax(1)

        _, idx_src2dest = linear_sum_assignment(distance_matrix(dest_norm, src_norm))
        return idx_src2dest
    
    def align_sdf(self, sdfs_src, feature_src, feature_dest):
        matching = self.object_matching(feature_src, feature_dest)
        sdfs_aligned = sdfs_src[matching]
        return sdfs_aligned

    def get_aligned_sdfs(self, img_src, img_dest):
        sdfs_src, _, features_src = self.get_sdf_features(img_src)
        sdfs_dest, _, features_dest = self.get_sdf_features(img_dest)
        matching = self.object_matching(features_src, features_dest)
        sdfs_aligned = sdfs_dest[matching]
        return (sdfs_src, sdfs_aligned)

