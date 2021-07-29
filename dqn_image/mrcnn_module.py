import os
import sys
import cv2
import numpy as np

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

class MaskRCNN():
    def __init__(self, scale=4):
        self.scale = scale

        self.config = InferenceConfig()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        set_session(tf.Session(config=tf_config))

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir='log', config=self.config)
        self._load_model(os.path.join(FILE_PATH, '../mask_rcnn'))


    def _load_model(self, model_path):
        COCO_MODEL_PATH = os.path.join(model_path, "mask_rcnn_coco.h5")

        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            mrcnn_utils.download_trained_weights(COCO_MODEL_PATH)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

    def get_segmentation(self, image):
        if image.shape[0]==3:
            image = image.transpose([1,2,0])
        results = self.model.detect([image], verbose=1)
        return results[0]

    def get_scaled_segmentation(self, image):
        if image.shape[0]==3:
            image = image.transpose([1,2,0])
        im_large = cv2.resize(image, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        results = self.model.detect([im_large], verbose=1)
        return results[0]

    def get_masks(self, image, scale_on=True):
        pad = 10
        image = np.pad(image[pad:-pad, pad:-pad], [[pad,pad],[pad,pad], [0, 0]], 'edge')
        if scale_on:
            seg_result = self.get_scaled_segmentation(image)
        else:
            seg_result = self.get_segmentation(image)
        masks = []
        colors = []
        feature_maps = []
        for m in range(len(seg_result['scores'])):
            mask = seg_result['masks'][:,:,m].astype(np.float32)
            mask_rgb = np.stack([mask] * 3)
            mask_rgb = mask_rgb.transpose([1,2,0])
            if scale_on:
                mask_resized = cv2.resize(mask_rgb, (0, 0), fx=1/self.scale, fy=1/self.scale)
                mask = mask_resized[:,:,0]
            masks.append(mask)

            color = image[mask.astype(bool)].mean(0)/255.
            colors.append(color)

            feature_map = seg_result['feature_maps'].max(1)[m]
            feature_maps.append(feature_map)

        return masks, colors, feature_maps
        
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

    def mask_over(self, image, threshold):
        return (image >= threshold).all(-1)

    def mask_under(self, image, threshold):
        return (image <= threshold).all(-1)

    def get_workspace_seg(self, image):
        return 1 - np.all([self.mask_over(image, [0.81, 0.92, 0.98]), 
                            self.mask_under(image, [0.86, 0.98, 1.])], 0)
