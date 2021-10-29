import cv2
import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt

ID = '############'

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device(ID)

profile = pipe.start(cfg)
for x in range(5): # skip 5 first frames
    pipe.wait_for_frames()

frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()
pipe.stop()

color = np.asanyarray(color_frame.get_data())
colorizer = rs.colorizer()
depth_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())

# alignment
align = rs.align(rs.stream.color)
frameset = align.process(frameset)

depth_frame_aligned = frameset.get_depth_frame()
depth_colorized = np.asanyarray(colorizer.colorize(depth_frame_aligned).get_data())

