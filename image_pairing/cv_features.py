import numpy as np
import sys
from random import randint

from cv_location import VisualOdometry2
from imagery_utils import SiftFeature, pose_realign
from pose_ana import *
import datetime
import pickle

#location = '/home/weihao/Projects/cambridge/OldHospital'
#pose_file = 'dataset_train.txt'
#poses_dic, cam = load_cambridge_poses(location, pose_file)

#
# location = '/home/weihao/Projects/kitty_dataset'
# pose_file = '00'
# poses_dic, cam = load_kitty_poses(location, pose_file)

c = 'heads'
mode = 'Train'
location = "/home/weihao/Projects/datasets/indoors/{}".format(c) #office" #heads
pose_file = "{}Split.txt".format(mode)
poses_dic, cam = load_indoor_7_poses(location, pose_file)
filename = '/home/weihao/Projects/p_files/{}_{}_cv.p'.format(c, mode)

print location, pose_file, filename

for p in poses_dic:
    print p, len(poses_dic[p])

sf = SiftFeature()
vo = VisualOdometry2(cam, sf)

w2 = cam.cx
h2 = cam.cy
t0 = datetime.datetime.now()
nt = 0
data = []
num_match = 5
mini_features_matches = 20
image_list = {}

length = 0
for seq in poses_dic:
    poses = poses_dic[seq]
    for id in poses:
        image_list[length] = (seq, id)
        length += 1

for id in image_list:
    cnt = 0
    while True:
        idx = randint(0, length-1)
        if idx != id:
            id1 = image_list[id]
            id2 = image_list[idx]
            vo.get_features(id, poses_dic[id1[0]][id1[1]], poses_dic[id2[0]][id2[1]])
            fs = vo.features
            if fs is not None and len(fs)>mini_features_matches:
                fs[:, 0] = (fs[:, 0] - w2) / w2
                fs[:, 1] = (fs[:, 1] - h2) / h2
                fs[:, 2] = (fs[:, 2] - w2) / w2
                fs[:, 3] = (fs[:, 3] - h2) / h2
                data.append([fs, vo.truth * 180 / np.pi])
            cnt += 1

        if cnt==num_match:
            break

    if id % 100 == 0:
        print id, len(data), datetime.datetime.now() - t0

print 'Total data', len(data)

with open(filename, 'w') as fp:
    pickle.dump(data, fp)