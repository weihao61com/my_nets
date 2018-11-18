import numpy as np
import sys
from random import randint, random

from cv_location import VisualOdometry2
from imagery_utils import SiftFeature, pose_realign
from pose_ana import *
import datetime
import pickle
import cv2
import matplotlib.pyplot as plt

filename = '/home/weihao/Projects/p_files/heads_Test_cv.p'
filename = '/home/weihao/Projects/p_files/kitty_02_Test_cv.p'

if len(sys.argv)>1:
    filename = sys.argv[1]

focal = 525.0
cam = PinholeCamera(640.0, 480.0, focal, focal, 320.0, 240.0)
w2 = cam.cx
h2 = cam.cy

output_file = '/home/weihao/tmp/feature.csv'
fp = open(output_file, 'w')

with open(filename, 'r') as f:
    data = pickle.load(f)

print len(data)
rs = []
nt = 0
for d in data:
    d0 = d[0]
    d1 = d[1]
    nt += 1
    if np.linalg.norm(d1)<4:
        continue
    print nt, d1
    plt.scatter(d0[:, 2] - d0[:,0], d0[:, 3]-d0[:, 1],  color='black', s=5)
    plt.show()
    # d0[:, 0] = d0[:, 0] * w2 + w2
    # d0[:, 1] = d0[:, 1] * h2 + h2
    # d0[:, 2] = d0[:, 2] * w2 + w2
    # d0[:, 3] = d0[:, 3] * h2 + h2
    # px_new = d0[:, :2]
    # px_last = d0[:, 2:]
    # E, mask = cv2.findEssentialMat(px_new, px_last, cameraMatrix=cam.mx,
    #                                method=cv2.RANSAC)
    # mh, R, t, mask0 = cv2.recoverPose(E, px_new, px_last, cameraMatrix=cam.mx)
    #
    # b = Utils.rotationMatrixToEulerAngles(R)
    # a = d[1]/180*np.pi
    # dr = a - b
    # r0 = np.linalg.norm(dr)* 180 / np.pi
    # fp.write('{},{},{},{},{},{},{}\n'.
    #          format(a[0], a[1], a[2], b[0], b[1], b[2], r0))
    # rs.append(r0)

print '\nmedian', np.median(rs)
fp.close()

