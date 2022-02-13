import os
import sys
import cv2
import copy
import torch
import skfmm
import numpy as np

from scipy.ndimage import morphology
from skimage.morphology import convex_hull_image
import torchvision.models as models

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

file_path = os.path.dirname(os.path.abspath(__file__))
# file_path = 'home/gun/Desktop/ur5_manipulation/object_wise/dqn'
sys.path.append(os.path.join(file_path, '../..', 'UnseenObjectClustering'))
import networks
from fcn.test_dataset import clustering_features #, test_sample
from fcn.config import cfg_from_file

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SDFModule():
    def __init__(self, rgb_feature=True, ucn_feature=False, resnet_feature=False, convex_hull=False, binary_hole=True):
        self.pretrained = os.path.join(file_path, '../..', 'UnseenObjectClustering', \
                'experiments/checkpoints/seg_resnet34_8s_embedding_cosine_color_sampling_epoch_16.checkpoint.pth')
        self.cfg_file = os.path.join(file_path, '../..', 'UnseenObjectClustering', \
                'experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml')
        cfg_from_file(self.cfg_file)

        self.network_name = 'seg_resnet34_8s_embedding'
        self.network_data = torch.load(self.pretrained)
        self.network = networks.__dict__[self.network_name](2, 64, self.network_data).to(device)
        self.network.eval()

        self.network_crop = None
        self.target_resolution = 96

        self.rgb_feature = rgb_feature
        self.ucn_feature = ucn_feature
        self.resnet_feature = resnet_feature
        if self.resnet_feature:
            self.resnet50 = models.resnet50(pretrained=True).to(device)

        self.convex_hull = convex_hull
        self.binary_hole = binary_hole

    def detect_objects(self, image, data_format='HWC'):
        if data_format=='HWC':
            image = image.transpose([2, 0, 1])
        im_tensor = torch.Tensor(image).unsqueeze(0).to(device)
        features = self.network(im_tensor, None).detach()
        out_label, selected_pixels = clustering_features(features[:1], num_seeds=100)

        segmap = out_label.cpu().detach().numpy()[0]
        return segmap

    def get_masks(self, image, data_format='HWC', rotate=False):
        if data_format=='HWC':
            image = image.transpose([2, 0, 1])
        if rotate:
            images = [image]
            for r in range(1, 4):
                im_rot = np.rot90(image, k=r, axes=(1, 2))
                images.append(im_rot.copy())
            images = np.array(images)
            im_tensor = torch.Tensor(images).to(device)
        else:
            im_tensor = torch.Tensor(image).unsqueeze(0).to(device)
        features = self.network(im_tensor, None).detach()
        out_label, selected_pixels = clustering_features(features[:1], num_seeds=100)

        features = features.cpu().numpy()
        if rotate:
            features_align = []
            for i, f in enumerate(features):
                f_reversed = np.rot90(f, k=-i, axes=(1, 2))
                features_align.append(f_reversed.copy())
            features = np.max(np.abs(features_align), axis=0)
        else:
            features = features[0]

        segmap = out_label.cpu().detach().numpy()[0]
        num_blocks = int(segmap.max())
        masks = []
        for nb in range(1, num_blocks+1):
            _mask = (segmap == nb).astype(float)
            masks.append(_mask)
        return masks, features

    def get_sdf(self, masks):
        sdfs = []
        for seg in masks:
            if seg.sum()==0:
                continue
            sd = skfmm.distance(seg.astype(int) - 0.5, dx=1)
            sdfs.append(sd)
        return np.array(sdfs) 

    def get_sdf_features(self, image, data_format='HWC', resize=True, rotate=False, clip=False):
        if len(image)==2:
            depth = image[1]
            image = image[0]
            if data_format=='HWC':
                image[:20] = [0.81960784, 0.93333333, 1.]
                image = image.transpose([2, 0, 1])
            depth_mask = (depth<0.9702)
            masks, features = self.get_masks(image, data_format='CHW', rotate=rotate)
            new_masks = []
            for m in masks:
                m = m * depth_mask
                if m.sum() < 30:
                    continue
                # convex hull
                if self.convex_hull:
                    m = convex_hull_image(m).astype(int)
                # binary hole filling
                elif self.binary_hole:
                    m = morphology.binary_fill_holes(m).astype(int)
                # check IoU with other masks
                duplicate = False
                for _m in new_masks:
                    intersection = np.all([m, _m], 0)
                    if intersection.sum() > min(m.sum(), _m.sum())/2:
                        duplicate = True
                        break
                if not duplicate:
                    new_masks.append(m)
            masks = new_masks
        else:
            if data_format=='HWC':
                image[:20] = [0.81960784, 0.93333333, 1.]
                image = image.transpose([2, 0, 1])
            masks, features = self.get_masks(image, data_format='CHW', rotate=rotate)
        sdfs = self.get_sdf(masks)
        if clip:
            sdfs = np.clip(sdfs, -100, 100)

        if self.rgb_feature or self.ucn_feature:
            rgb_features = []
            ucn_features = []
            for sdf in sdfs:
                local_rgb = image[:, sdf>=0].mean(1)
                local_feature = features[:, sdf>=0].mean(1)
                rgb_features.append(local_rgb)
                ucn_features.append(local_feature)
            rgb_features = np.array(rgb_features)
            ucn_features = np.array(ucn_features)
        if self.resnet_feature:
            segmented_images = []
            for sdf in sdfs:
                segment_image = copy.deepcopy(image)
                segment_image[:, sdf<=0] = 0.
                segmented_images.append(segment_image)
            inputs = torch.Tensor(segmented_images).to(device)
            if len(inputs.shape)!=4:
                resnet_features = np.array([])
            else:
                resnet_features = self.resnet50(inputs).cpu().detach().numpy()

        block_features = []
        if self.rgb_feature:
            block_features.append(rgb_features)
        if self.ucn_feature:
            block_features.append(ucn_features)
        if self.resnet_feature:
            block_features.append(resnet_features)

        if resize:
            res = self.target_resolution
            sdfs_resized = []
            for sdf in sdfs:
                resized = cv2.resize(sdf, (res, res), interpolation=cv2.INTER_AREA)
                sdfs_resized.append(resized)
            sdfs_raw = sdfs
            sdfs = np.array(sdfs_resized)/400.

        return sdfs, sdfs_raw, block_features
    
    def object_matching(self, features_src, features_dest, use_cnn=False):
        if len(features_src[0])==0 or len(features_dest[0])==0:
            idx_src2dest = np.array([], dtype=int)
            return idx_src2dest

        if self.resnet_feature:
            features_src[-1] /= 5.
            features_dest[-1] /= 5.
        concat_src = np.concatenate(features_src, 1)
        concat_dest = np.concatenate(features_dest, 1)

        src_norm = concat_src / np.linalg.norm(concat_src, axis=1).reshape(len(concat_src), 1)
        dest_norm = concat_dest / np.linalg.norm(concat_dest, axis=1).reshape(len(concat_dest), 1)
        idx_dest, idx_src = linear_sum_assignment(distance_matrix(dest_norm, src_norm))
        return idx_dest, idx_src
        
    def align_sdf(self, matching, sdf_src, sdf_target):
        aligned = np.zeros_like(sdf_target)
        # detection fail
        if len(matching)<2:
            return aligned
        aligned[matching[0]] = sdf_src[matching[1]]
        return aligned

    def get_aligned_sdfs(self, img_src, img_dest):
        sdfs_src, _, features_src = self.get_sdf_features(img_src)
        sdfs_dest, _, features_dest = self.get_sdf_features(img_dest)
        matching = self.object_matching(features_src, features_dest)
        sdfs_aligned = sdfs_dest[matching]
        return (sdfs_src, sdfs_aligned)

    def oracle_align(self, sdfs, pixel_poses, scale=5):
        N = len(pixel_poses)
        if len(sdfs)==0:
            return np.zeros([N, 96, 96])
        H, W = sdfs[0].shape
        aligned = np.zeros([N, H, W])

        centers = []
        for sdf in sdfs:
            mx, my = np.where(sdf==sdf.max())
            centers.append([mx.mean(), my.mean()])
        centers = scale * np.array(centers)
        idx_p, idx_c = linear_sum_assignment(distance_matrix(pixel_poses, centers))
        aligned[idx_p] = sdfs[idx_c]
        return aligned
