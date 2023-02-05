import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '..'))

import argparse
from matplotlib import pyplot as plt
from backgroundsubtraction_module import *

from functools import partial
from jgcpd import RigidPartRegistration

parser = argparse.ArgumentParser()
parser.add_argument("--res", default=480, type=int)
parser.add_argument("--rgb", action="store_true")
parser.add_argument("--hsv", action="store_true")
args = parser.parse_args()

def visualize3(iteration, error, X, Y, Z, ax1, ax2):
    ax1.cla()
    ax2.cla()
    ax2.set_yticks([])
    TY = Y[np.where(Z)[1], np.where(Z)[0]]
    ax1.set_title('Target: X', loc='center')
    ax2.set_title('Source: T(Y)', loc='center')
    if X.shape[1] > 2:
        ax1.scatter(X[:, 0], -X[:, 1], c=X[:, 2:5], marker=".")
        ax2.scatter(X[:, 0], -X[:, 1], c='red', label='Target', marker=".")
        ax2.scatter(TY[:, 0], -TY[:, 1], c=TY[:, 2:5], marker=".")
    else:
        ax1.scatter(X[:, 0],  -X[:, 1], marker=".") #, label='Target')
        ax2.scatter(X[:, 0],  -X[:, 1], c='red', label='Target', marker=".")
        ax2.scatter(TY[:, 0],  -TY[:, 1], marker=".")#, label='Source')

    xmin = min(TY[:, 0].min(), X[:, 0].min()) - 0.1
    xmax = max(TY[:, 0].max(), X[:, 0].max()) + 0.1
    ymin = -max(TY[:, 1].max(), X[:, 1].max()) - 0.1
    ymax = -min(TY[:, 1].min(), X[:, 1].min()) + 0.1
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    plt.text(0.82, 0.07, 'Iteration: {:d}\nQ: {:06.4f}'.format(iteration, np.mean(error)), 
             horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, 
             fontsize='x-large')
    ax2.legend(loc='upper left', fontsize='x-large')
    #plt.draw()
    #plt.pause(0.001)

res = args.res
RGB = args.rgb
HSV = args.hsv
if HSV:
    RGB = True

for img_idx in range(100):
    regist = 'rigid' # affine / rigid / deformable
    backsub = BackgroundSubtraction(res=res)
    backsub.fitting_model()

    rgb_s = np.array(Image.open('state/%d.png'%img_idx))
    hsv_s = cv2.cvtColor(cv2.cvtColor(rgb_s, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    m_s = backsub.get_mask(rgb_s)

    m_s_resized = cv2.resize(m_s, (res, res), interpolation=cv2.INTER_NEAREST)
    rgb_s_resized = cv2.resize(rgb_s, (res, res), interpolation=cv2.INTER_AREA)
    hsv_s_resized = cv2.cvtColor(cv2.cvtColor(rgb_s_resized, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)

    y, x = np.where(m_s_resized)
    points_s = np.concatenate([np.array([x/res-0.5]).T, np.array([y/res-0.5]).T, rgb_s_resized[y, x]/255., hsv_s_resized[y, x]/255.], 1)

    rgb_g = np.array(Image.open('goal/%d.png'%img_idx))
    m_g, cm_g ,fm_g = backsub.get_masks(rgb_g)

    num_obj = len(m_g)
    points_g = []
    z_g = []

    rgb_g_resized = cv2.resize(rgb_g, (res, res), interpolation=cv2.INTER_AREA)
    hsv_g_resized = cv2.cvtColor(cv2.cvtColor(rgb_g_resized, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    for i, m in enumerate(m_g):
        m_resized = cv2.resize(m, (res, res), interpolation=cv2.INTER_NEAREST)
        y, x = np.where(m_resized)

        p = np.concatenate([np.array([x/res-0.5]).T, np.array([y/res-0.5]).T, rgb_g_resized[y, x]/255., hsv_g_resized[y, x]/255.], 1)
        points_g.append(p)

        z = np.zeros([len(p), num_obj])
        z[:, i] = 1.
        z_g.append(z)

    points_g = np.concatenate(points_g, 0)
    z_g = np.concatenate(z_g, 0)

    fig_vis = plt.figure(figsize=(10, 5))
    fig_vis.add_axes([0, 0, 0.5, 1])
    fig_vis.add_axes([0.5, 0, 0.5, 1])
    callback = partial(visualize3, ax1=fig_vis.axes[0], ax2=fig_vis.axes[1])
    remove_colors = False
    if remove_colors:
        points_s[:, 2:] = [0, 0, 0]
        points_g[:, 2:] = [0, 0, 0]

    print("===================== Result ========================")
    reg = RigidPartRegistration(**{'X': points_s, 'Y': points_g, 'Z': z_g, 'RGB': RGB, 'HSV': HSV, 'K': num_obj})
    TY, (R, t) = reg.register(callback)
    #print(TY[0])
    #print(R[0], t[0])
    print("Q value:", reg.q)
    print("=====================================================")
    if HSV:
        fig_vis.savefig('results/iter_hsv_%d.png'%img_idx)
    else:
        fig_vis.savefig('results/iter_rgb_%d.png'%img_idx)


    TY_combined = TY[np.where(z_g)[1], np.where(z_g)[0]]
    fig = plt.figure(figsize=(10, 5))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    ax0.scatter(points_s[:, 0], -points_s[:, 1], c=points_s[:, 2:5], marker=".")
    ax1.scatter(TY_combined[:, 0], -TY_combined[:, 1], c=points_g[:, 2:5], marker=".")

    xmin = min(TY_combined[:, 0].min(), points_s[:, 0].min()) - 0.01
    xmax = max(TY_combined[:, 0].max(), points_s[:, 0].max()) + 0.01
    ymin = -max(TY_combined[:, 1].max(), points_s[:, 1].max()) - 0.01
    ymax = -min(TY_combined[:, 1].min(), points_s[:, 1].min()) + 0.01
    ax0.set_xlim(xmin, xmax)
    ax1.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)
    ax1.set_ylim(ymin, ymax)
    if HSV:
        fig.savefig('results/result_hsv_%d.png'%img_idx)
    else:
        fig.savefig('results/result_rgb_%d.png'%img_idx)

