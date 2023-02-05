from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py import MjRenderContextOffscreen
import mujoco_py

import copy
import cv2
import glfw
from matplotlib import pyplot as plt
import numpy as np
import imageio
import types
import time

import xml.etree.ElementTree as ET
from collections import OrderedDict
from base import MujocoXML

import os
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '../../pointcloud'))
from pcd_gen import *

def new_joint(**kwargs):
    element = ET.Element("joint", attrib=kwargs)
    return element

def array_to_string(array):
    return " ".join(["{}".format(x) for x in array])

def string_to_array(string):
    return np.array([float(x) for x in string.split(" ")])


class Tabletop(MujocoXML):
    def __init__(self):
        super().__init__(os.path.join(file_path, 'make_urdf/meshes_mujoco/tabletop.xml'))
        #self.set_sensor()

    def set_sensor(self):
        sensor = ET.SubElement(self.root, 'sensor')
        f1 = ET.SubElement(sensor, 'force')
        f2 = ET.SubElement(sensor, 'force')
        f1.set('name', 'left_finger_force')
        f1.set('site', 'left_inner_finger_sensor')
        f2.set('name', 'right_finger_force')
        f2.set('site', 'right_inner_finger_sensor')


class MujocoXMLObject(MujocoXML):
    def __init__(self, fname):
        MujocoXML.__init__(self, fname)

    def get_bottom_offset(self):
        bottom_site = self.worldbody.find("./body/site[@name='bottom_site']")
        return string_to_array(bottom_site.get("pos"))

    def get_top_offset(self):
        top_site = self.worldbody.find("./body/site[@name='top_site']")
        return string_to_array(top_site.get("pos"))

    def get_horizontal_radius(self):
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='horizontal_radius_site']"
        )
        return string_to_array(horizontal_radius_site.get("pos"))[0]

    def get_collision(self, name=None):

        collision = copy.deepcopy(self.worldbody.find("./body/body[@name='collision']"))
        collision.attrib.pop("name")
        if name is not None:
            collision.attrib["name"] = name
            geoms = collision.findall("geom")
            if len(geoms) == 1:
                geoms[0].set("name", name)
            else:
                for i in range(len(geoms)):
                    geoms[i].set("name", "{}-{}".format(name, i))
        return collision

    def get_visual(self, name=None, site=False):
        visual = copy.deepcopy(self.worldbody.find("./body/body[@name='visual']"))
        visual.attrib.pop("name")
        if name is not None:
            visual.attrib["name"] = name
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
            visual.append(ET.Element("site", attrib=template))
        return visual


class TabletopTask(Tabletop):
    def __init__(self, mujoco_objects):
        """
        mujoco_objects: a list of MJCF models of physical objects
        """
        super().__init__()

        # temp: z-rotation
        self.z_rotation = True

        self.merge_objects(mujoco_objects)
        self.set_objects_geom(mass=0.02)
        self.save = False

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        self.objects = []  # xml manifestation
        self.max_horizontal_radius = 0
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name)
            obj.append(new_joint(name=obj_name, type="hinge", damping="0.0005"))
            self.objects.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

    def set_objects_geom(self, mass=0.02):
        for o in self.objects:
            o.find('geom').set('mass', f'{mass}')
            o.find('geom').set('friction', "0.1 0.1 0.5")
            o.find('geom').set('solimp', "0.9 0.95 0.001")
            o.find('geom').set('solref', "0.001 1.0")

    def sample_quat(self):
        """Samples quaternions of random rotations along the z-axis."""
        if self.z_rotation:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
            return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]
        return [1, 0, 0, 0]

    def random_quat(self, rand=None):
        """Return uniform random unit quaternion.
        rand: array like or None
            Three independent random variables that are uniformly distributed
            between 0 and 1.
        >>> q = random_quat()
        >>> np.allclose(1.0, vector_norm(q))
        True
        >>> q = random_quat(np.random.random(3))
        >>> q.shape
        (4,)
        """
        if rand is None:
            rand = np.random.rand(3)
        else:
            assert len(rand) == 3
        r1 = np.sqrt(1.0 - rand[0])
        r2 = np.sqrt(rand[0])
        pi2 = np.pi * 2.0
        t1 = pi2 * rand[1]
        t2 = pi2 * rand[2]
        return np.array(
            (np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2),
            dtype=np.float32,
        )    

    def place_objects(self):
        """Places objects randomly until no collisions or max iterations hit."""
        placed_objects = []
        index = 0
        # place objects by rejection sampling
        for _, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            #print('horizontal_radius', horizontal_radius)
            #print('bottom_offset', bottom_offset)
            success = False
            for _ in range(5000):  # 5000 retries
                object_z = np.random.uniform(high=1.0, low=1.4)
                #bin_x_half = self.bin_size[0] / 2.0 - horizontal_radius - (self.bin_size[2] - object_z) - 0.02
                #bin_y_half = self.bin_size[1] / 2.0 - horizontal_radius - (self.bin_size[2] - object_z) - 0.02
                object_x = np.random.uniform(high=0.2, low=-0.2)
                object_y = np.random.uniform(high=0.3, low=-0.1)

                # make sure objects do not overlap
                pos = np.array([object_x, object_y, object_z])
                location_valid = True
                for pos2, r in placed_objects:
                    dist = np.linalg.norm(pos[:2] - pos2[:2], np.inf)
                    if dist <= 0.02: #r + horizontal_radius:
                        location_valid = False
                        break

                # place the object
                if location_valid:
                    # add object to the position
                    placed_objects.append((pos, horizontal_radius))
                    self.objects[index].set("pos", array_to_string(pos))
                    # random z-rotation
                    #quat = self.sample_quat()
                    quat = self.random_quat()
                    self.objects[index].set("quat", array_to_string(quat))
                    success = True
                    break

            # raise error if all objects cannot be placed after maximum retries
            if not success:
                raise Exception #RandomizationError("Cannot place all objects in the bins")
            index += 1

    def place_single_objects(self, index):
        placed_objects = []
        obj_list = []
        for _, obj_mjcf in self.mujoco_objects.items():
            obj_list.append(obj_mjcf)
        obj_mjcf = obj_list[index]
        horizontal_radius = obj_mjcf.get_horizontal_radius()
        bottom_offset = obj_mjcf.get_bottom_offset()
        object_z = np.random.uniform(high=self.bin_size[2], low=0.02)
        bin_x_half = self.bin_size[0] / 2.0 - horizontal_radius - (self.bin_size[2] - object_z) - 0.02
        bin_y_half = self.bin_size[1] / 2.0 - horizontal_radius - (self.bin_size[2] - object_z) - 0.02
        object_x = np.random.uniform(high=bin_x_half, low=-bin_x_half)
        object_y = np.random.uniform(high=bin_y_half, low=-bin_y_half)

        # make sure objects do not overlap
        object_xyz = np.array([object_x, object_y, object_z])
        pos = self.bin_offset - bottom_offset + object_xyz

        self.objects[index].set("pos", array_to_string(pos))
        quat = self.random_quat()
        self.objects[index].set("quat", array_to_string(quat))
        index += 1

    def place_col_objects(self):
        placed_objects = []
        index = 0

        for _, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            object_z = np.random.uniform(high=self.bin_size[2] + 0.5, low=self.bin_size[2])
            bin_x_half = self.bin_size[0] / 2 - horizontal_radius - 0.05
            bin_y_half = self.bin_size[1] / 2 - horizontal_radius - 0.05
            object_x = np.random.uniform(high=bin_x_half, low=-bin_x_half)
            object_y = np.random.uniform(high=bin_y_half, low=-bin_y_half)
            object_xyz = np.array([object_x, object_y, object_z])
            pos = self.bin_offset - bottom_offset + object_xyz
            placed_objects.append((pos, horizontal_radius))
            self.objects[index].set("pos", array_to_string(pos))
            quat = self.sample_quat()
            self.objects[index].set("quat", array_to_string(quat))
            index += 1


class TabletopEnv():
    def __init__(
            self, 
            render=True,
            camera_height=64,
            camera_width=64,
            control_freq=8,
            data_format='NHWC',
            camera_name='rlview2',
            gpu=-1,
            dataset='train1',
            small=False
            ):

        self.real_object = True
        self.dataset = dataset
        self.small = small
        self.render = render
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.control_freq = control_freq
        self.data_format = data_format
        self.camera_name = camera_name
        self.gpu = gpu

        mujoco_objects = self.load_objects(num=0)
        self.object_names = list(mujoco_objects.keys())
        self.num_objects = len(self.object_names)
        self.selected_objects = list(range(self.num_objects))

        self.model = TabletopTask(mujoco_objects) 
        self.model.place_objects()
        self.mjpy_model = self.model.get_model(mode='mujoco_py')
        #self.model_xml = 'make_urdf/meshes_mujoco/tabletop.xml'
        #self.model = load_model_from_path(os.path.join(file_path, self.model_xml))
        self.n_substeps = 1  # 20

        self.set_sim()
        self.sim.forward()
        if self.render: 
            self.sim.render(mode='window')

    def load_objects(self, num=0):
        if self.dataset=="test":
            obj_list = []
            for o in range(12):
                if self.small:
                    obj_list.append('small-test%d'%o)
                else:
                    obj_list.append('test%d'%o)

        elif self.dataset=="train1":
            obj_list = []
            for o in range(16):
                if self.small:
                    obj_list.append('small-train1-%d'%o)
                else:
                    obj_list.append('train1-%d'%o)

        elif self.dataset=="train2":
            obj_list = []
            for o in range(15):
                if self.small:
                    obj_list.append('small-train2-%d'%o)
                else:
                    obj_list.append('train2-%d'%o)

        self.obj_list = obj_list
        obj_dirpath = 'make_urdf/objects/'
        obj_counts = [0] * len(obj_list)
        lst = []

        if num==0:
            num = len(obj_list)
        for n in range(num):
            #print("spawning %d"%n, obj_list[n])
            rand_obj = n
            #rand_obj = np.random.randint(len(obj_list))
            obj_name = obj_list[rand_obj]
            obj_count = obj_counts[rand_obj]
            obj_xml = MujocoXMLObject(os.path.join(file_path, obj_dirpath, '%s.xml'%obj_name))
            lst.append(("%s_%d"%(obj_name, obj_count), obj_xml))
            obj_counts[rand_obj] += 1
        mujoco_objects = OrderedDict(lst)
        return mujoco_objects

    def select_objects(self, num=3, idx=-1):
        if idx==-1:
            indices = np.random.choice(range(self.num_objects), num, False)
        else:
            indices = np.arange(num*idx, num*(idx+1)) % 32
        self.selected_objects = list(indices)

    def destroy_viewer(self):
        glfw.destroy_window(self.viewer.window)
        self.viewer = None

    def reset_viewer(self):
        if self.viewer is not None:
            if self.render:
                glfw.destroy_window(self.viewer.window)
            else:
                glfw.destroy_window(self.viewer.opengl_context.window)
            self.viewer = None
        self.set_sim()
        self._init_robot()
        self.sim.forward()

    def set_sim(self):
        self.sim = MjSim(self.mjpy_model, nsubsteps=self.n_substeps)
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

    def calculate_depth(self, depth):
        zNear = 0.01
        zFar = 50
        return zNear / (1 - depth * (1 - zNear / zFar))

    def get_obs(self):
        if self.render: 
            self.sim.render(mode='window')

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

if __name__=='__main__':
    env = TabletopEnv()
    for i in range(10):
        rgb, depth = env.get_obs()
        plt.imshow(rgb)
        plt.show()

        PCG = PointCloudGen()
        pcd = PCG.pcd_from_rgbd(rgb, depth)
        PCG.visualize(pcd)
