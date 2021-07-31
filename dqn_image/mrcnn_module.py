import os
import sys
import cv2
import numpy as np

from backgroundsubtraction_module import BackgroundSubtraction

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../mask_rcnn'))

import mrcnn_utils
import model as modellib
import visualize
import coco

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRCNN(BackgroundSubtraction):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

        self.config = InferenceConfig()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        set_session(tf.Session(config=tf_config))

        # Create model object in inference mode.
        self.mrcnn = modellib.MaskRCNN(mode="inference", model_dir='log', config=self.config)
        self._load_model(os.path.join(FILE_PATH, '../mask_rcnn'))


    def _load_model(self, model_path):
        COCO_MODEL_PATH = os.path.join(model_path, "mask_rcnn_coco.h5")

        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            mrcnn_utils.download_trained_weights(COCO_MODEL_PATH)

        # Load weights trained on MS-COCO
        self.mrcnn.load_weights(COCO_MODEL_PATH, by_name=True)

    def get_segmentation(self, image):
        if image.shape[0]==3:
            image = image.transpose([1,2,0])
        results = self.mrcnn.detect([image], verbose=0)
        return results[0]

    def get_scaled_segmentation(self, image):
        if image.shape[0]==3:
            image = image.transpose([1,2,0])
        im_large = cv2.resize(image, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        results = self.mrcnn.detect([im_large], verbose=0)
        return results[0]

    def get_masks_with_contours(self, fmask):
        fmask = fmask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(fmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        masks = []
        for ns in range(len(contours)):
            zeros = np.zeros_like(fmask)
            obj_mask = cv2.drawContours(zeros, contours, ns, 1, -1)
            masks.append(obj_mask)
        return masks

    def get_masks(self, image, scale_on=False):
        pad = self.pad
        image = np.pad(image[pad:-pad, pad:-pad], [[pad, pad], [pad, pad], [0, 0]], 'edge').astype(np.uint8)

        fmask = self.model.apply(image).astype(bool)
        mask_bgsub = self.get_masks_with_contours(fmask)

        image_black = np.zeros_like(image)
        image_white = np.ones_like(image) * 255
        image_black[fmask] = image[fmask]
        image_white[fmask] = image[fmask]

        if scale_on:
            seg_black = self.get_scaled_segmentation(image_black)
            seg_white = self.get_scaled_segmentation(image_white)
            # seg_result = self.get_scaled_segmentation(image)
        else:
            seg_black = self.get_segmentation(image_black)
            seg_white = self.get_segmentation(image_white)
            # seg_result = self.get_segmentation(image)

        mask_black = []
        for i in range(seg_black['masks'].shape[-1]):
            fmask = seg_black['masks'][:, :, i].astype(np.uint8)
            contours, hierarchy = cv2.findContours(fmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for ns in range(len(contours)):
                zeros = np.zeros_like(fmask)
                obj_mask = cv2.drawContours(zeros, contours, ns, 1, -1)
                mask_black.append(obj_mask)
        mask_white = []
        for i in range(seg_white['masks'].shape[-1]):
            fmask = seg_white['masks'][:, :, i].astype(np.uint8)
            contours, hierarchy = cv2.findContours(fmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for ns in range(len(contours)):
                zeros = np.zeros_like(fmask)
                obj_mask = cv2.drawContours(zeros, contours, ns, 1, -1)
                mask_white.append(obj_mask)

        mask_list = sorted(mask_black + mask_white, key=lambda m: m.sum()) + mask_bgsub
        mask_list = [m for m in mask_list if m.sum()>30]
        if len(mask_list)==0:
            print("!!")
        mask_selected = [mask_list[0]]
        for mask in mask_list[1:]:
            check_duplicate = False
            for idx, ms in enumerate(mask_selected):
                mask_union = np.any([mask, ms], 0)
                mask_intersection = np.all([mask, ms], 0)
                if mask_intersection.sum() / mask_union.sum() > 0.5:
                    check_duplicate = True
                    if mask.sum() < ms.sum():
                        mask_selected[idx] = mask
            if not check_duplicate:
                mask_selected.append(mask)

        # resize the masks
        if scale_on:
            for idx, mask in enumerate(mask_selected):
                mask_rgb = np.stack([mask] * 3)
                mask_rgb = mask_rgb.transpose([1, 2, 0])
                mask_resized = cv2.resize(mask_rgb, (0, 0), fx=1 / self.scale, fy=1 / self.scale)
                mask_selected[idx] = mask_resized[:, :, 0]

        # get segmented colors
        colors = []
        for mask in mask_selected:
            color = image[mask.astype(bool)].mean(0) / 255.
            colors.append(color)

        return mask_selected, colors
        
    def visualize(self, image, result):
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

        visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'],
                            class_names, result['scores'])
