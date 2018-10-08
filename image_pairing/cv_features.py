import numpy as np
import sys

from cv_location import VisualOdometry2
from utils import SiftFeature, pose_realign
from pose_ana import *
import datetime
import pickle

range1 = 5
range2 = 20

#location = '/home/weihao/Projects/cambridge/OldHospital'
#pose_file = 'dataset_train.txt'
#poses_dic, cam = load_cambridge_poses(location, pose_file)

#
# location = '/home/weihao/Projects/kitty_dataset'
# pose_file = '00'
# poses_dic, cam = load_kitty_poses(location, pose_file)

c = 'pumpkin'
location = "/home/weihao/Projects/indoor/{}".format(c) #office" #heads
pose_file = "TrainSplit.txt"
poses_dic, cam = load_indoor_7_poses(location, pose_file)
filename = '/home/weihao/Projects/p_files/{}_tr{}_2.p'.format(c, range2)


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
for seq in poses_dic:
    #poses = pose_realign(poses_dic[seq])
    poses = poses_dic[seq]
    for img_id1 in poses:
        #if len(data)>50:
        #    break
        for img_id2 in poses:
            if range1 < abs(img_id2-img_id1) < range2:
                vo.get_features(img_id1, poses[img_id1], poses[img_id2])
                fs = vo.features
                if fs is not None:
                    fs[:, 0] = (fs[:, 0]-w2)/w2
                    fs[:, 1] = (fs[:, 1]-h2)/h2
                    fs[:, 2] = (fs[:, 2]-w2)/w2
                    fs[:, 3] = (fs[:, 3]-h2)/h2
                    data.append([fs, vo.truth*180/np.pi])
        nt += 1
        if nt % 100 == 0:
            print nt, img_id1, len(data), datetime.datetime.now() - t0

print 'Total data', len(data)

with open(filename, 'w') as fp:
    pickle.dump(data, fp)