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

mini_match = 20
mini_range = 500
top_matches = 20
diff=1

key = 'office'
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

filename = '{}/p_files/{}_{}_r_{}.p'.format(HOME, key, mode, diff)
output_file = '{}/tmp/{}_{}.csv'.format(HOME, key, mode)
print location, filename, output_file

for p in poses_dic:
    print p, len(poses_dic[p])

sf = SiftFeature()
vo = VisualOdometry2(cam, sf, mini_match)

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
#image_list = {}

length = 0
for s1 in poses_dic:
    poses1 = poses_dic[s1]
    for id1 in poses1:
        pose1 = poses1[id1]
        fs_list = []

        for s2 in poses_dic:
            poses2 = poses_dic[s1]
            for id2 in poses2:
                pose2 = poses1[id2]
                if abs(s1-s2)!=diff:
                    continue

                if s1==s2 and abs(id1-id2)<mini_range:
                    continue

                vo.get_features_2(id1, pose1, pose2)
                fs = vo.features
                if len(fs) > mini_match:
                    fs[:, 0] = (fs[:, 0]-w2)/w2
                    fs[:, 1] = (fs[:, 1]-h2)/h2
                    fs[:, 2] = (fs[:, 2]-w2)/w2
                    fs[:, 3] = (fs[:, 3]-h2)/h2
                    fs[:, 4] = (fs[:, 4]-w2)/w2
                    fs[:, 5] = (fs[:, 5]-h2)/h2
                    fs_list.append([fs, vo.truth])

        fs_list.sort(key=lambda x: x[0].shape[0], reverse=True)

        for fs in fs_list[:top_matches]:
            data.append(fs)

        nt += 1
        if nt % 100 == 0:
            print nt, s1, s2, id1, id2, len(data), datetime.datetime.now() - t0
            t0 = datetime.datetime.now()

print 'Total data', len(data)

with open(filename, 'w') as fp:
    pickle.dump(data, fp)