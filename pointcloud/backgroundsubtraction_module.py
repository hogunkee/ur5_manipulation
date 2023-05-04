import os
import cv2
import numpy as np
from PIL import Image
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

from sklearn.cluster import SpectralClustering

class BackgroundSubtraction():
    def __init__(self, res=96):
        self.pad = 10
        self.res = res
        self.model = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        # self.fitting_model()

        self.workspace_seg = None
        # self.make_empty_workspace_seg()
        
    def fitting_model_from_data(self, data_path):
        pad = self.pad
        data_list = os.listdir(data_path)
        for data in data_list:
            if not data.endswith('.png'):
                continue
            frame = np.array(Image.open(os.path.join(data_path, data)))
            frame = np.pad(frame[pad:-pad, pad:-pad], [[pad, pad], [pad, pad], [0, 0]], 'edge').astype(np.uint8)
            if frame.shape[1] != self.res:
                frame = cv2.resize(frame, (self.res, self.res), interpolation=cv2.INTER_AREA)
            self.model.apply(frame)

    def load_images(self, data_dir='.'):
        images = []
        states = sorted(os.listdir(os.path.join(data_dir, 'state')))
        for s_img in states:
            s_path = os.path.join(data_dir, 'state', s_img)
            images.append(np.array(Image.open(s_path)))
        
        goals = sorted(os.listdir(os.path.join(data_dir, 'goal')))
        for g_img in goals:
            g_path = os.path.join(data_dir, 'goal', g_img)
            images.append(np.array(Image.open(g_path)))
        return images                          

    def fitting_model(self):
        self.model = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        pad = self.pad
        frames = self.load_images('.')

#         frame = (np.load(os.path.join(FILE_PATH, '../dqn_image/scenes/bg.npy')) * 255).astype(np.uint8)
#         frame = np.pad(frame[pad:-pad, pad:-pad], [[pad, pad], [pad, pad], [0, 0]], 'edge')
#         self.model.apply(frame)

#         frames = (np.load(os.path.join(FILE_PATH, '../dqn_image/scenes/rgb.npy')) * 255).astype(np.uint8)
#         frames = np.pad(frames[:, pad:-pad, pad:-pad], [[0,0], [pad,pad], [pad,pad], [0,0]], 'edge')
        for frame in frames:
            self.model.apply(frame)

    def get_points(self, image):
        pad = self.pad 
        #image = np.pad(image[pad:-pad, pad:-pad], [[pad,pad],[pad,pad], [0, 0]], 'edge').astype(np.uint8)
        fmask = self.model.apply(image, 0, 0)

        my, mx = np.nonzero(fmask)
        points = np.array(list(zip(mx, my)))
        return points

        im_blur = cv2.blur(image, (5, 5))
        colors = np.array([im_blur[y, x] / (255) for x, y in zip(mx, my)])

        points = np.concatenate([points/self.res - 0.5, colors], 1)
        return points

    def get_mask(self, image):
        pad = self.pad 
        image = np.pad(image[pad:-pad, pad:-pad], [[pad,pad],[pad,pad], [0, 0]], 'edge').astype(np.uint8)
        fmask = self.model.apply(image, 0, 0)
        return fmask

    def get_masks(self, image, n_cluster=3):
        image = np.copy((image*255).astype(np.uint8))
        pad = self.pad
        image = np.pad(image[pad:-pad, pad:-pad], [[pad,pad],[pad,pad], [0, 0]], 'edge').astype(np.uint8)
        if image.shape[1] != self.res:
            image = cv2.resize(image, (self.res, self.res), interpolation=cv2.INTER_AREA)
        fmask = self.model.apply(image, 0, 0)

        my, mx = np.nonzero(fmask)
        points = list(zip(mx, my, np.ones_like(mx) * self.res))
        z = (np.array(points).T / np.linalg.norm(points, axis=1)).T

        im_blur = cv2.blur(image, (5, 5))
        colors = np.array([im_blur[y, x] / (10 * 255) for x, y in zip(mx, my)])
        z_color = np.concatenate([z, colors], 1)
        clusters = SpectralClustering(n_clusters=n_cluster, n_init=10).fit_predict(z_color)

        new_mask = np.zeros([fmask.shape[0], fmask.shape[1], n_cluster])
        for x, y, c in zip(mx, my, clusters):
            new_mask[y, x, c] = 1
        masks = new_mask.transpose([2,0,1]).astype(float)

        # # Opening
        # for i in range(len(masks)):
        #     masks[i] = cv2.morphologyEx(masks[i], cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        colors = []
        for mask in masks:
            color = image[mask.astype(bool)].mean(0) / 255.
            colors.append(color)

        return masks, np.array(colors), fmask

        # contours, hierarchy = cv2.findContours(fmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # masks, colors = [], []
        # num_seg = len(contours)
        # for ns in range(num_seg):
        #     zeros = np.zeros_like(fmask)
        #     obj_mask = cv2.drawContours(zeros, contours, ns, 1, -1)
        #     obj_color = image[obj_mask.astype(bool)].mean(0)/255.
        #     masks.append(obj_mask)
        #     colors.append(obj_color)
        # return np.array(masks), np.array(colors), fmask, contours

    def mask_over(self, image, threshold):
        return (image >= threshold).all(-1)

    def mask_under(self, image, threshold):
        return (image <= threshold).all(-1)

    def get_workspace_seg(self, image):
        pad = self.pad
        image = np.pad(image[pad:-pad, pad:-pad], [[pad,pad],[pad,pad], [0, 0]], 'edge')/255
        return self.mask_over(image, [.97, .97, .97])

    def make_empty_workspace_seg(self):
        pad = self.pad
        frame = (np.load(os.path.join(FILE_PATH, '../dqn_image/scenes/bg.npy')) * 255).astype(np.uint8)
        frame = np.pad(frame[pad:-pad, pad:-pad], [[pad, pad], [pad, pad], [0, 0]], 'edge')
        self.workspace_seg = self.get_workspace_seg(frame)
