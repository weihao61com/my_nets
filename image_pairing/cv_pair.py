import numpy as np
import sys

from cv_location import VisualOdometry2
# from visual_odometry import VisualOdometry
from imagery_utils import SiftFeature, pose_realign
# from utils import Utils
from pose_ana import *
import datetime
import pickle

project_dir = '/home/weihao/Projects'

range1 = 15
range2 = 15
#location = '/home/weihao/Projects/cambridge/OldHospital'
#pose_file = '/dataset_test.txt'
#poses_dic, cam = load_cambridge_poses(location, pose_file)

#
#location = '/home/weihao/Projects/kitty_dataset'
#pose_file = '00'
#poses_dic, cam = load_kitty_poses(location, pose_file)
project_dir = '/home/weihao/Projects'
key = 'heads'  #office" #heads
mode = 'Train'
location = "{}/datasets/indoors/{}".format(project_dir, key)
pose_file = "{}Split.txt".format(mode)
poses_dic, cam = load_indoor_7_poses(location, pose_file)
output_file = '{}/tmp/{}_{}_{}.csv'.format(project_dir, key, mode, range2)
filename = '{}/p_files/{}_{}_{}_{}_2.p'.format(project_dir, mode, key, range1, range2)

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
rs = []
with open(output_file, 'w') as fp:
    for seq in poses_dic:
        #poses = pose_realign(poses_dic[seq])
        poses = poses_dic[seq]
        for img_id1 in poses:
            #if img_id1 >100:
            #    break
            for img_id2 in poses:
                if range1 <= abs(img_id2-img_id1) <=range2:
                    vo.process(img_id1, poses[img_id1], img_id2, poses[img_id2])
                    b = Utils.rotationMatrixToEulerAngles(vo.R)
                    a = Utils.rotationMatrixToEulerAngles(vo.pose_R)
                    c = Utils.rotationMatrixToEulerAngles(poses[img_id1].m3x3)
                    d = poses[img_id1].tran
                    matches += vo.matches
                    inline += vo.inline
                    dr = a-b
                    # for x in range(len(dr)):
                    #     if dr[x]>np.pi:
                    #         dr[x] = np.pi*2-dr[x]
                    #     if dr[x]<-np.pi:
                    #         dr[x] = np.pi*2+dr[x]
                    fs = vo.features
                    if fs is not None:
                        r0 = np.linalg.norm(dr)*180/np.pi
                        fp.write('{},{},{},{},{},{},{},{},{},{},'
                                 '{},{},{},{},{},{},{},{},{},{}\n'.
                                 format(seq,
                            img_id1, img_id2, vo.matches, vo.inline, \
                            a[0], a[1], a[2], b[0], b[1], b[2], \
                            c[0], c[1], c[2], d[0], d[1], d[2], r0, vo.m1, vo.m2
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

                    nt += 1
                    if nt%100==0:
                        print nt, img_id1, datetime.datetime.now() - t0


print "count", nt, float(matches)/nt, float(inline)/nt
print "output", output_file
print 'median', np.median(rs)

if filename is not None:
    with open(filename, 'w') as fp:
        pickle.dump(data, fp)