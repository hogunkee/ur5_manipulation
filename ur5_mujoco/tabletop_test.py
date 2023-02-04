from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py import MjRenderContextOffscreen
import mujoco_py

import cv2
import glfw
from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
import imageio
import types
import time

import os
file_path = os.path.dirname(os.path.abspath(__file__))

class TabletopEnv():
    def __init__(
            self, 
            render=True,
            image_state=True,
            camera_height=64,
            camera_width=64,
            control_freq=8,
            data_format='NHWC',
            camera_name='rlview',
            gpu=-1,
            testset=False
            ):
        self.model_xml = 'make_urdf/meshes_mujoco/tabletop.xml'

        self.real_object = False
        self.render = render
        self.image_state = image_state
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.control_freq = control_freq
        self.data_format = data_format
        self.camera_name = camera_name
        self.gpu = gpu

        self.object_names = ['target_body_%d'%d for d in range(15)]
        #self.object_names = ['target_body_%d'%(d+1) for d in range(6)]
        self.num_objects = len(self.object_names)
        self.selected_objects = list(range(self.num_objects))

        self.model = load_model_from_path(os.path.join(file_path, self.model_xml))
        # self.model = load_model_from_path(os.path.join(file_path, 'make_urdf/ur5_robotiq.xml'))
        self.n_substeps = 1  # 20
        self.sim = MjSim(self.model, nsubsteps=self.n_substeps)
        if self.render:
            self.viewer = MjViewer(self.sim)
            self.viewer._hide_overlay = True
            # Camera pose
            lookat_refer = [0., 0., 0.9]  # self.sim.data.get_body_xpos('target_body_1')
            self.viewer.cam.lookat[0] = lookat_refer[0]
            self.viewer.cam.lookat[1] = lookat_refer[1]
            self.viewer.cam.lookat[2] = lookat_refer[2]
            self.viewer.cam.azimuth = -90 #0 # -65 #-75 #-90 #-75
            self.viewer.cam.elevation = -60  # -30 #-60 #-15
            self.viewer.cam.distance = 2.0  # 1.5
        else:
            if self.gpu==-1:
                self.viewer = MjRenderContextOffscreen(self.sim)
            else:
                self.viewer = MjRenderContextOffscreen(self.sim, self.gpu)

        self.sim.forward()

    def calculate_depth(self, depth):
        zNear = 0.01
        zFar = 50
        return zNear / (1 - depth * (1 - zNear / zFar))

    def get_obs(self):
        if self.render:
            self.viewer._set_mujoco_buffers()
            self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, depth=True, mode='offscreen')
            camera_obs = self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, depth=True, mode='offscreen')
            im_rgb, im_depth = camera_obs
            self.viewer._set_mujoco_buffers()

        else:
            self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, mode='offscreen')
            camera_obs = self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, depth=True, mode='offscreen')
            im_rgb, im_depth = camera_obs

        im_rgb = np.flip(im_rgb, axis=1) / 255.0
        if self.data_format=='NCHW':
            im_rgb = np.transpose(im_rgb, [2, 0, 1])

        im_depth = self.calculate_depth(np.flip(im_depth, axis=1))
        return im_rgb, im_depth

