import copy
import numpy as np
import open3d as o3d

from matplotlib import pyplot as plt
from PIL import Image

class PointCloudGen(object):
    def __init__(self):
        self.set_camera(fovy=45, width=480, height=480)
        self.z_threshold = 0.333

    def set_camera(self, fovy=45, width=480, height=480):
        self.f = 0.5 * height / np.tan(fovy * np.pi / 360)
        self.K = np.array([[self.f, 0, width/2],
                      [0, self.f, height/2],
                      [0, 0, 1]])
        return

    def pcd_from_depth(self, depth_image):
        height, width = depth_image.shape
        pcd = []
        for i in range(height):
            for j in range(width):
                if i<40 and 120<j<240:
                    continue
                if depth_image[i, j] > self.z_threshold:
                    continue
                z = 0.6 - depth_image[i, j]
                x = (j - width / 2) * z / self.f
                y = (height / 2 - i) * z / self.f
                pcd.append([x, y, z])
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        return pcd_o3d
    
    def pcd_from_rgbd(self, rgb_image, depth_image):
        height, width = depth_image.shape
        pcd = []
        colors = []
        for i in range(height):
            for j in range(width):
                if i<40 and 120<j<240:
                    continue
                if depth_image[i, j] > self.z_threshold:
                    continue
                z = 0.6 - depth_image[i, j]
                x = (j - width/2) * z / self.f
                y = (height/2 - i) * z / self.f
                pcd.append([x, y, z])
                colors.append(rgb_image[i, j])
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
        return pcd_o3d

    def pcd_from_pointset(self, pointset, colors=None):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pointset)
        if colors:
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
        return pcd_o3d

    def visualize(self, pcd):
        o3d.visualization.draw_geometries([pcd])


if __name__=='__main__':
    depth_image = np.load('goal/100.npy')
    rgb_image = np.array(Image.open('goal/100.png'))

    pcg = PointCloudGen()
    pcd_d = pcg.pcd_from_depth(depth_image)
    pcd_rgbd = pcg.pcd_from_rgbd(rgb_image, depth_image)
    pcg.visualize(pcd_d)
    pcg.visualize(pcd_rgbd)
