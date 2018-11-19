import numpy as np
import sys
from random import randint, random

from cv_location import VisualOdometry2
from imagery_utils import SiftFeature, pose_realign
from pose_ana import *
import datetime
import pickle

num_match = 50
key = 'heads'
mode = 'Train'

if len(sys.argv)>1:
    key = sys.argv[1]
if len(sys.argv)>2:
    mode = sys.argv[2]

print key, mode

#location = '/home/weihao/Projects/cambridge/OldHospital'
#pose_file = 'dataset_train.txt'
#poses_dic, cam = load_cambridge_poses(location, pose_file)

#
# location = '/home/weihao/Projects/kitty_dataset'
# pose_file = '00'
# poses_dic, cam = load_kitty_poses(location, pose_file)


location = "/home/weihao/Projects/datasets/indoors/{}".format(key) #office" #heads
pose_file = "{}Split.txt".format(mode)
poses_dic, cam = load_indoor_7_poses(location, pose_file)
filename = '/home/weihao/Projects/p_files/{}_{}_cv_m{}.p'.format(key, mode, num_match)

print location, pose_file, filename
isTest = mode.startswith('Te')
fp = None
if isTest:
    output_file = '/home/weihao/tmp/{}_{}.csv'.format(key, mode)
    fp = open(output_file, 'w')

for p in poses_dic:
    print p, len(poses_dic[p])

sf = SiftFeature()
vo = VisualOdometry2(cam, sf)

w2 = cam.cx
h2 = cam.cy
t0 = datetime.datetime.now()
nt = 0
data = []
mini_features_matches = 50
image_list = {}
rs = []

length = 0
for seq in poses_dic:
    poses = poses_dic[seq]
    for id in poses:
        image_list[length] = (seq, id)
        length += 1

for id in image_list:
    cnt = 0
    idxs = range(length)
    np.random.shuffle(idxs)
    for x in range(length):
        idx = idxs[x]
        if idx != id:
            id1 = image_list[id]
            id2 = image_list[idx]
            vo.get_features(id, poses_dic[id1[0]][id1[1]], poses_dic[id2[0]][id2[1]], isTest)
            fs = vo.features
            if fs is not None and len(fs)>mini_features_matches:
                fs[:, 0] = (fs[:, 0] - w2) / w2
                fs[:, 1] = (fs[:, 1] - h2) / h2
                fs[:, 2] = (fs[:, 2] - w2) / w2
                fs[:, 3] = (fs[:, 3] - h2) / h2
                data.append([fs, vo.truth * 180 / np.pi])
                cnt += 1

                if isTest:
                    pose1 = poses_dic[id1[0]][id1[1]]
                    pose2 = poses_dic[id2[0]][id2[1]]

                    b = Utils.rotationMatrixToEulerAngles(vo.R)
                    a = Utils.rotationMatrixToEulerAngles(vo.pose_R)
                    c = Utils.rotationMatrixToEulerAngles(pose1.m3x3)
                    d = pose1.tran
                    dr = a - b
                    r0 = np.linalg.norm(dr) * 180 / np.pi
                    if random() < 0.1:
                        fp.write('{},{},{},{},{},{},{},{},{},'
                                 '{},{},{},{},{},{},{},{},{},{}\n'.
                                 format(pose1.filename, pose2.filename,
                                        vo.matches, vo.inline,
                                        a[0], a[1], a[2], b[0], b[1], b[2],
                                        c[0], c[1], c[2], d[0], d[1], d[2],
                                        r0, vo.m1, vo.m2
                                        ))
                    rs.append(r0)

        if cnt==num_match:
            break

    if id % 100 == 0:
        print id, len(data), datetime.datetime.now() - t0

print 'Total data', len(data), len(rs)

with open(filename, 'w') as fp:
    pickle.dump(data, fp)

if isTest:
    fp.close()
    print '\nmedian', key, np.median(rs)

