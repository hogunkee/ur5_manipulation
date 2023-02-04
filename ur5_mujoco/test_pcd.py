from open3d import camera, geometry
from ur5_env import *

width = 512
height = 512

env = UR5Env(camera_height=height, camera_width=width, xml_ver=0, camera_depth=True, gpu=-1)
env.move_to_pos()
# place objects #
x = np.linspace(-0.3, 0.3, 5)
y = np.linspace(0.4, -0.2, 5)
xx, yy = np.meshgrid(x, y, sparse=False)
xx = xx.reshape(-1)
yy = yy.reshape(-1)

for obj_idx in range(15): #16
    env.sim.data.qpos[7 * obj_idx + 12: 7 * obj_idx + 15] = [xx[obj_idx], yy[obj_idx], 0.9]
    print(obj_idx, xx[obj_idx], yy[obj_idx])
env.sim.forward()

fovy = env.sim.model.cam_fovy[0]
f = 0.5 * height / np.tan(fovy * np.pi / 360)
#K = np.array([[-f, 0, width/2], [0, f, height/2], [0, 0, 1]])

cx = width//2
cy = height//2
intrinsic = camera.PinholeCameraIntrinsic(width=width, height=height, fx=f, fy=f, cx=cx, cy=cy)

rgb, depth = env.move_to_pos(get_img=True)

rim = geometry.Image((rgb*255).astype(np.uint8))
dim = geometry.Image((depth*1000).astype(np.uint8))
rgbd_im = geometry.RGBDImage.create_from_color_and_depth(rim, dim, convert_rgb_to_intensity=False)

pcd = geometry.PointCloud.create_from_rgbd_image(rgbd_im, intrinsic=intrinsic)
#pcd = geometry.PointCloud.create_from_depth_image(dim, intrinsic=intrinsic)

