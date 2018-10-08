import numpy as np
import sys

from cv_location import VisualOdometry2
# from visual_odometry import VisualOdometry
from utils import SiftFeature, pose_realign
from pose_ana import *
import datetime

range1 = 5
range2 = 20
#location = '/home/weihao/Projects/cambridge/OldHospital'
#pose_file = '/dataset_test.txt'
#poses_dic, cam = load_cambridge_poses(location, pose_file)

#
#location = '/home/weihao/Projects/kitty_dataset'
#pose_file = '00'
#poses_dic, cam = load_kitty_poses(location, pose_file)

location = "/home/weihao/Projects/indoor/chess" #office" #heads
pose_file = "TestSplit.txt"
poses_dic, cam = load_indoor_7_poses(location, pose_file)
output_file = '/home/weihao/tmp/chess_te20.csv'

print location, pose_file

for p in poses_dic:
    print p, len(poses_dic[p])

sf = SiftFeature()
vo = VisualOdometry2(cam, sf)

t0 = datetime.datetime.now()
nt = 0
matches = 0
inline = 0
with open(output_file, 'w') as fp:
    for seq in poses_dic:
        #poses = pose_realign(poses_dic[seq])
        poses = poses_dic[seq]
        for img_id1 in poses:
            for img_id2 in poses:
                if range1 < img_id2-img_id1 <range2:
                    vo.process(img_id1, poses[img_id1], img_id2, poses[img_id2])
                    b = rotationMatrixToEulerAngles(vo.R)
                    a = rotationMatrixToEulerAngles(vo.pose_R)
                    c = rotationMatrixToEulerAngles(poses[img_id1].m3x3)
                    d = poses[img_id1].tran
                    matches += vo.matches
                    inline += vo.inline
                    dr = a-b
                    for x in range(len(dr)):
                        if dr[x]>np.pi:
                            dr[x] = np.pi*2-dr[x]
                        if dr[x]<-np.pi:
                            dr[x] = np.pi*2+dr[x]
                    r0 = np.linalg.norm(dr)*180/np.pi
                    #print img_id1, img_id2, vo.matches, \
                    #    a[0], a[1], a[2], b[0], b[1], b[2],\
                    #    c[0], c[1], c[2], d[0], d[1], d[2]
                    fp.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.
                             format(seq,
                        img_id1, img_id2, vo.matches, vo.inline, \
                        a[0], a[1], a[2], b[0], b[1], b[2], \
                        c[0], c[1], c[2], d[0], d[1], d[2], r0, vo.m1, vo.m2
                    ))
                    nt += 1
                    if nt%1000==0:
                        print nt, img_id1, datetime.datetime.now() - t0


print "count", nt, float(matches)/nt, float(inline)/nt
print "output", output_file