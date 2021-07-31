import os
import cv2
import numpy as np
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

class BackgroundSubtraction():
    def __init__(self):
        self.pad = 10
        self.model = None
        self.sub_model = None
        self.fitting_model()
        # self.fitting_submodel()

        self.workspace_seg = None
        self.make_empty_workspace_seg()

    def fitting_model(self):
        self.model = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        pad = self.pad 
        frame = (np.load(os.path.join(FILE_PATH, 'scenes/bg.npy')) * 255).astype(np.uint8)
        frame = np.pad(frame[pad:-pad, pad:-pad], [[pad, pad], [pad, pad], [0, 0]], 'edge')
        self.model.apply(frame)

        frames = (np.load(os.path.join(FILE_PATH, 'scenes/rgb.npy')) * 255).astype(np.uint8)
        frames = np.pad(frames[:, pad:-pad, pad:-pad], [[0,0], [pad,pad], [pad,pad], [0,0]], 'edge')
        for frame in frames:
            self.model.apply(frame)

    def fitting_submodel(self):
        self.sub_model = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        pad = self.pad
        frame = (np.load(os.path.join(FILE_PATH, 'scenes/bg.npy')) * 255).astype(np.uint8)
        frame = np.pad(frame[pad:-pad, pad:-pad], [[pad, pad], [pad, pad], [0, 0]], 'edge')
        self.sub_model.apply(frame)

        frames = (np.load(os.path.join(FILE_PATH, 'scenes/rgb.npy')) * 255).astype(np.uint8)
        frames = np.pad(frames[:, pad:-pad, pad:-pad], [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'edge')
        for frame in np.random.permutation(frames):
            self.sub_model.apply(frame)

    def get_masks(self, image, sub=False):
        if sub:
            model = self.sub_model
        else:
            model = self.model
        pad = self.pad 
        image = np.pad(image[pad:-pad, pad:-pad], [[pad,pad],[pad,pad], [0, 0]], 'edge').astype(np.uint8)
        fmask = model.apply(image)
        contours, hierarchy = cv2.findContours(fmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        masks, colors = [], []
        num_seg = len(contours)
        for ns in range(num_seg):
            zeros = np.zeros_like(fmask)
            obj_mask = cv2.drawContours(zeros, contours, ns, 1, -1)
            obj_color = image[obj_mask.astype(bool)].mean(0)/255.
            masks.append(obj_mask)
            colors.append(obj_color)
        return np.array(masks), np.array(colors), fmask, contours

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
        frame = (np.load(os.path.join(FILE_PATH, 'scenes/bg.npy')) * 255).astype(np.uint8)
        frame = np.pad(frame[pad:-pad, pad:-pad], [[pad, pad], [pad, pad], [0, 0]], 'edge')
        self.workspace_seg = self.get_workspace_seg(frame)
