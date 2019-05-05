import numpy as np
import sys
from random import randint, random

from cv_location import VisualOdometry2
from imagery_utils import SiftFeature, pose_realign
from pose_ana import *
import datetime
import pickle
import cv2

filename = '/Users/weihao/Projects/p_files/heads_Test_cv_s3.p'
if len(sys.argv)>1:
    filename = sys.argv[1]

if 'kitty' in filename:
    focal = 719  # 719
    cam = PinholeCamera(1241.0, 376.0, focal, focal, 607.1928, 185.2157)
else:
    focal = 525.0
    cam = PinholeCamera(640.0, 480.0, focal, focal, 320.0, 240.0)

#focal = 719  # 719
#cam = PinholeCamera(1241.0, 376.0, focal, focal, 607.1928, 185.2157)

w2 = cam.cx
h2 = cam.cy

output_file = '/home/weihao/tmp/feature.csv'
fp = open(output_file, 'w')

with open(filename, 'r') as f:
    data = pickle.load(f)

print len(data)
rs = []
angs = []
rs0 = []
rs1 = []
rs2 = []
for d in data:

    d0 = d[0]
    d0[:, 0] = d0[:, 0] * w2 + w2
    d0[:, 1] = d0[:, 1] * h2 + h2
    d0[:, 2] = d0[:, 2] * w2 + w2
    d0[:, 3] = d0[:, 3] * h2 + h2
    px_new = d0[:, :2]
    px_last = d0[:, 2:4]
    if d0.shape[1]>4:
        d1 = d0[:,6]
        d2 = d0[:,7]
    E, mask = cv2.findEssentialMat(px_new, px_last, cameraMatrix=cam.mx,
                                   method=cv2.RANSAC)
    mh, R, t, mask0 = cv2.recoverPose(E, px_new, px_last, cameraMatrix=cam.mx)

    b = Utils.rotationMatrixToEulerAngles(R)*180/np.pi
    a = d[1][:3]
    dr = a - b
    r0 = np.linalg.norm(dr)
    rs.append(r0)

    if random() > 1000.0/len(data):
        continue

    fp.write('{},{},{},{},{},{},{}\n'.
             format(a[0], a[1], a[2], b[0], b[1], b[2], r0))
    rs0.append(abs(dr[0]))
    rs1.append(abs(dr[1]))
    rs2.append(abs(dr[2]))
    angs.append(np.linalg.norm(a))

print 'name, median, Anger-error, mx, my, mz'
print '{0}, {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}'.format(
    os.path.basename(filename), np.median(rs), np.median(angs),
    np.median(rs0),np.median(rs1),np.median(rs2))
fp.close()

rs = sorted(rs)
length = len(rs)
fp = open(output_file + '.csv', 'w')
for a in range(length):
    fp.write('{},{}\n'.format(rs[a], float(a) / length))
fp.close()