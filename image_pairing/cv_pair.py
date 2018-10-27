import numpy as np
import sys
from random import randint

from cv_location import VisualOdometry2
# from visual_odometry import VisualOdometry
from imagery_utils import SiftFeature, pose_realign
# from utils import Utils
from pose_ana import *
import datetime
import pickle

project_dir = '/home/weihao/Projects'
key = 'heads'
mode = 'Test'

if len(sys.argv)>1:
    key = sys.argv[1]
if len(sys.argv)>2:
    mode = sys.argv[2]

#range1 = 2
#range2 = 2
#location = '/home/weihao/Projects/cambridge/OldHospital'
#pose_file = '/dataset_test.txt'
#poses_dic, cam = load_cambridge_poses(location, pose_file)

#
# location = '{}/datasets/kitty'.format(project_dir)
# key = '02'
# mode = 'Test'
# poses_dic, cam = load_kitty_poses(location, key)

location = "/home/weihao/Projects/datasets/indoors/{}".format(key) #office" #heads
pose_file = "{}Split.txt".format(mode)
poses_dic, cam = load_indoor_7_poses(location, pose_file)

output_file = '{}/tmp/{}_{}.csv'.format(project_dir, key, mode)
filename = '{}/p_files/{}_{}_cv.p'.format(project_dir, mode, key)

print location, pose_file, 'focal', cam.fx

for p in poses_dic:
    print p, len(poses_dic[p])

sf = SiftFeature()
vo = VisualOdometry2(cam, sf)

w2 = cam.cx
h2 = cam.cy
t0 = datetime.datetime.now()
nt = 0
matches = 0
inline = 0
data = []

rs= []

num_match = 40
mini_features_matches = 100
image_list = {}

length = 0
for seq in poses_dic:
    poses = poses_dic[seq]
    for id in poses:
        image_list[length] = (seq, id)
        length += 1

fp = None
if mode.startswith('Te'):
    fp = open(output_file, 'w')


for id in image_list:
    cnt = 0
    while True:
        idx = randint(0, length-1)
        if idx != id:
            id1 = image_list[id]
            id2 = image_list[idx]
            pose1 = poses_dic[id1[0]][id1[1]]
            pose2 = poses_dic[id2[0]][id2[1]]
            vo.get_features(id, pose1, pose2, fp is not None)
            fs = vo.features

            if len(fs)>=mini_features_matches:

                if fp:
                    b = Utils.rotationMatrixToEulerAngles(vo.R)
                    a = Utils.rotationMatrixToEulerAngles(vo.pose_R)
                    c = Utils.rotationMatrixToEulerAngles(pose1.m3x3)
                    d = pose1.tran
                    matches += vo.matches
                    inline += vo.inline
                    dr = a-b
                    r0 = np.linalg.norm(dr)*180/np.pi
                    fp.write('{},{},{},{},{},{},{},{},{},'
                             '{},{},{},{},{},{},{},{},{},{}\n'.
                             format(pose1.filename,pose2.filename,
                             vo.matches, vo.inline,
                             a[0], a[1], a[2], b[0], b[1], b[2],
                             c[0], c[1], c[2], d[0], d[1], d[2],
                                    r0, vo.m1, vo.m2
                    ))
                    rs.append(r0)

                a0 = (fs[:, 0] - w2) / w2
                a1 = (fs[:, 1] - h2) / h2
                a2 = (fs[:, 2] - w2) / w2
                a3 = (fs[:, 3] - h2) / h2
                ns= np.zeros(fs.shape)
                ns[:,0] = a0-a2
                ns[:,1] = a1-a3
                ns[:,2] = a0+a2
                ns[:,3] = a1+a3
                #if img_id1 == 0:
                #    for b in range(len(ns)):
                #        print ns[b]
                # data.append([ns, vo.truth * 180 / np.pi])
                data.append([ns, vo.truth * 180 / np.pi])
                # if img_id1==0:
                #     for a in range(len(vo.mask1)):
                #         print ns[a,0], ns[a,1], ns[a,2], ns[a,3],\
                #             vo.mask1[a][0], vo.mask2[a][0]

        cnt += 1
        if cnt>=num_match:
            break

    if id%100==0:
        print id, len(data), datetime.datetime.now() - t0
        t0 = datetime.datetime.now()

if fp:
    fp.close()
    mn = np.linalg.norm(rs)
    print 'rms, median', mn*mn/len(rs), np.median(rs)
    print 'match point', float(matches)/length, float(inline)/length

print "count", length, len(data)
print "output", output_file
if filename is not None:
    with open(filename, 'w') as fp:
        pickle.dump(data, fp)