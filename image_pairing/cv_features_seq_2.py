import numpy as np
import sys
from random import randint, random

from cv_location import VisualOdometry2
from imagery_utils import SiftFeature, pose_realign
from pose_ana import *
import datetime
import pickle

this_file_path = os.path.dirname(os.path.realpath(__file__))
HOME = '{}/../../'.format(this_file_path)

#HOME = '/home/weihao/Projects'

range2 = 1
range3 = 0
range1 = 0 #-range2
#key = 'heads'
#mode = 'Train'
key = '00'
mode = 'Train'

if len(sys.argv)>1:
    key = sys.argv[1]
if len(sys.argv)>2:
    mode = sys.argv[2]

print key, mode

#location = '/home/weihao/Projects/cambridge/OldHospital'
#pose_file = 'dataset_train.txt'
#poses_dic, cam = load_cambridge_poses(location, pose_file)

if key.startswith('0'):
    location = '{}/datasets/kitti'.format(HOME)
    poses_dic, cam = load_kitty_poses(location, key)
    key = 'kitti_{}'.format(key)
else:
    location = "{}/datasets/indoors/{}".format(HOME, key) #office" #heads
    poses_dic, cam = load_indoor_7_poses(location, "{}Split.txt".format(mode))

filename = '{}/p_files/{}_{}_cv_s{}_2.p'.format(HOME, key, mode, range2)
output_file = '{}/tmp/{}_{}.csv'.format(HOME, key, mode)
print location, filename, output_file

for p in poses_dic:
    print p, len(poses_dic[p])

sf = SiftFeature()
vo = VisualOdometry2(cam, sf)

w2 = cam.cx
h2 = cam.cy
t0 = datetime.datetime.now()
nt = 0
data = []
rs = []

isTest = mode.startswith('Te')
fp = None
if isTest:
    fp = open(output_file, 'w')

#num_match = 5
mini_features_matches = 20
#image_list = {}

length = 0
for seq in poses_dic:
    #poses = pose_realign(poses_dic[seq])
    poses = poses_dic[seq]
    for img_id1 in poses:
        # if len(data)>50:
        #    break
        pose1 = poses[img_id1]
        for img_id2 in poses:
            if abs(img_id2-img_id1)<=range3:
                continue
            if range1 <= img_id2-img_id1 <= range2:
                pose2 = poses[img_id2]
                vo.get_features_2(img_id1, pose1, pose2)
                fs = vo.features
                if len(fs) > mini_features_matches:
                    fs[:, 0] = (fs[:, 0]-w2)/w2
                    fs[:, 1] = (fs[:, 1]-h2)/h2
                    fs[:, 2] = (fs[:, 2]-w2)/w2
                    fs[:, 3] = (fs[:, 3]-h2)/h2
                    fs[:, 4] = (fs[:, 4]-w2)/w2
                    fs[:, 5] = (fs[:, 5]-h2)/h2
                    data.append([fs, vo.truth])

        nt += 1
        if nt % 100 == 0:
            print nt, img_id1, len(data), datetime.datetime.now() - t0
            t0 = datetime.datetime.now()

print 'Total data', len(data)

with open(filename, 'w') as fp:
    pickle.dump(data, fp)