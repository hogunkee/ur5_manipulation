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

from sklearn.cluster import SpectralClustering
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
    def __init__(self, rgb_feature=True, ucn_feature=False, resnet_feature=False, convex_hull=False, binary_hole=True, using_depth=False):
        self.using_depth = using_depth
        if self.using_depth:
            self.pretrained = os.path.join(file_path, '../..', 'UnseenObjectClustering', \
                    'experiments/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth')
            self.cfg_file = os.path.join(file_path, '../..', 'UnseenObjectClustering', \
                    'experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml')
        else:
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

        self.params = self.get_camera_params()
        self.rgb_feature = rgb_feature
        self.ucn_feature = ucn_feature
        self.resnet_feature = resnet_feature
        if self.resnet_feature:
            self.resnet50 = models.resnet50(pretrained=True).to(device)

        self.fdim = 0
        if self.rgb_feature:
            self.fdim += 3
        if self.ucn_feature:
            self.fdim += 64
        if self.resnet_feature:
            self.fdim += 1000

        self.convex_hull = convex_hull
        self.binary_hole = binary_hole

    def get_camera_params(self):
        params = {}
        params['img_width'] = 480
        params['img_height'] = 480

        fovy = 45
        f = 0.5 * params['img_height'] / np.tan(fovy * np.pi / 360)
        params['fx'] = f
        params['fy'] = f
        return params

    def detect_objects(self, rgb, depth, data_format='HWC'):
        if data_format=='HWC':
            rgb = rgb.transpose([2, 0, 1])
        rgb_tensor = torch.Tensor(rgb).unsqueeze(0).to(device)

        if self.using_depth:
            xyz_img = self.process_depth(depth)
            depth_tensor = torch.from_numpy(xyz_img).permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            depth_tensor = None

        features = self.network(rgb_tensor, None, depth_tensor).detach()
        out_label, selected_pixels = clustering_features(features[:1], num_seeds=100)

        segmap = out_label.cpu().detach().numpy()[0]
        return segmap

    def compute_xyz(self, depth):
        if 'fx' in self.params and 'fy' in self.params:
            fx = self.params['fx']
            fy = self.params['fy']
        else:
            aspect_ratio = self.params['img_width'] / self.params['img_height']
            e = 1 / (np.tan(np.radians(self.params['fov']/2.)))
            t = self.params['near'] / e
            b = -t
            r = t* aspect_ratio 
            l = -r
            alpha = self.params['img_width'] / (r - l)
            focal_length = self.params['near'] * alpha
            fx = focal_length
            fy = focal_length

        if 'x_offset' in self.params and 'y_offset' in self.params:
            x_offset = self.params['x_offset']
            y_offset = self.params['y_offset']
        else:
            x_offset = self.params['img_width'] / 2
            y_offset = self.params['img_height'] / 2

        indices = np.indices((self.params['img_height'], self.params['img_width']), dtype=np.float32).transpose(1,2,0)
        z_e = depth
        x_e = (indices[..., 1] - x_offset) * z_e / fx
        y_e = (indices[..., 0] - y_offset) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # shape: [H, W, 3]

        return xyz_img

    def process_depth(self, depth):
        data_augmentation = False
        if data_augmentation:
            pass
            #depth = augmentation.add_noise_to_depth(depth, self.params)
            #depth = augmentation.dropout_random_ellipses(depth, self.params)
        xyz_img = self.compute_xyz(depth)
        if data_augmentation:
            pass
            #xyz_img = augmantation.add_noise_to_xyz(xyz_img, depth, self.params)
        return xyz_img

    def get_masks(self, rgb, depth, data_format='HWC', rotate=False):
        if data_format=='HWC':
            rgb = rgb.transpose([2, 0, 1])
        if rotate:
            rgbs = [rgb]
            for r in range(1, 4):
                im_rot = np.rot90(rgb, k=r, axes=(1, 2))
                rgbs.append(im_rot.copy())
            rgbs = np.array(rgbs)
            rgb_tensor = torch.Tensor(rgbs).to(device)
        else:
            rgb_tensor = torch.Tensor(rgb).unsqueeze(0).to(device)

        if self.using_depth:
            xyz_img = self.process_depth(depth)
            depth_tensor = torch.Tensor(xyz_img).permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            depth_tensor = None

        features = self.network(rgb_tensor, None, depth_tensor).detach()
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

    def get_sdf_features(self, rgb, depth, nblock, data_format='HWC', resize=True, rotate=False, clip=False):
        if depth is not None:
            if data_format=='HWC':
                rgb[:20] = [0.81960784, 0.93333333, 1.]
                rgb = rgb.transpose([2, 0, 1])
            depth_mask = (depth<0.9702).astype(float)
            masks, latents = self.get_masks(rgb, depth, data_format='CHW', rotate=rotate)

            if resize:
                res = self.target_resolution
                new_masks = []
                for i in range(len(masks)):
                    new_masks.append(cv2.resize(masks[i], (res, res), interpolation=cv2.INTER_AREA))
                masks = np.array(new_masks)
                latents = cv2.resize(latents.transpose([1,2,0]), (res, res), interpolation=cv2.INTER_AREA).transpose([2,0,1])
                rgb = cv2.resize(rgb.transpose([1,2,0]), (res, res), interpolation=cv2.INTER_AREA).transpose([2,0,1])
                depth_mask = cv2.resize(depth_mask, (res, res), interpolation=cv2.INTER_AREA)
                depth_mask = depth_mask.astype(bool)

            # Spectral Clustering #
            masks = masks[:nblock].astype(bool)
            if len(masks) < nblock and np.sum(masks)!=0:
                use_rgb = True
                use_ucn_feature = True
                my, mx = np.nonzero(np.sum(masks, 0))
                points = list(zip(mx, my, np.ones_like(mx) * rgb.shape[1]))
                z = (np.array(points).T / np.linalg.norm(points, axis=1)).T
                if use_rgb:
                    point_colors = np.array([rgb[:, y, x] / (10*255) for x, y in zip(mx, my)])
                    z = np.concatenate([z, point_colors], 1)
                if use_ucn_feature:
                    point_ucnfeatures = np.array([latents[:, y, x] for x, y in zip(mx, my)])
                    z = np.concatenate([z, point_ucnfeatures], 1)
                clusters = SpectralClustering(n_clusters=nblock, n_init=10).fit_predict(z)
                sp_masks = np.zeros([nblock, rgb.shape[1], rgb.shape[2]])
                for x, y, c in zip(mx, my, clusters):
                    sp_masks[c, y, x] = 1
                masks = sp_masks

            # Depth Processing #
            depth_masks = []
            for m in masks:
                m = m * depth_mask
                if m.sum() < 10: #30
                    continue
                # convex hull
                if self.convex_hull:
                    m = convex_hull_image(m).astype(int)
                # binary hole filling
                elif self.binary_hole:
                    m = morphology.binary_fill_holes(m).astype(int)
                # check IoU with other masks
                duplicate = False
                for _m in depth_masks:
                    intersection = np.all([m, _m], 0)
                    if intersection.sum() > min(m.sum(), _m.sum())/2:
                        duplicate = True
                        break
                if not duplicate:
                    depth_masks.append(m)
            masks = depth_masks
        else:
            if data_format=='HWC':
                rgb[:20] = [0.81960784, 0.93333333, 1.]
                rgb = rgb.transpose([2, 0, 1])
            masks, latents = self.get_masks(rgb, None, data_format='CHW', rotate=rotate)
        sdfs = self.get_sdf(masks)
        if clip:
            sdfs = np.clip(sdfs, -100, 100)

        if self.rgb_feature or self.ucn_feature:
            rgb_features = []
            ucn_features = []
            for sdf in sdfs:
                local_rgb = rgb[:, sdf>=0].mean(1)
                local_feature = latents[:, sdf>=0].mean(1)
                rgb_features.append(local_rgb)
                ucn_features.append(local_feature)
            rgb_features = np.array(rgb_features)
            ucn_features = np.array(ucn_features)

        if self.resnet_feature:
            segmented_images = []
            for sdf in sdfs:
                segment_image = copy.deepcopy(rgb)
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
            sdfs_raw = []
            for sdf in sdfs:
                resized = cv2.resize(sdf, (5*res, 5*res), interpolation=cv2.INTER_AREA)
                sdfs_raw.append(resized)
            sdfs_raw = np.array(sdfs_raw)

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
