import numpy as np
import sys
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))
HOME = '{}/../..'.format(this_file_path)
sys.path.append('{}/my_nets'.format(HOME))

from image_pairing.pose_ana import \
    load_kitti_poses, load_indoor_7_poses, load_TUM_poses
from dataset_h import load_kitti_poses

from utils import Utils


def find_time(t, time_table):
    idx0 = 0
    idx1 = len(time_table)-1

    while idx1>idx0+1:
        idx = int((idx1+idx0)/2)
        if time_table[idx]>t:
            idx1 = idx
        elif time_table[idx]<t:
            idx0 = idx
        else:
            return idx, 0

    if t-time_table[idx0] < time_table[idx1]-t:
        return idx0, (t-time_table[idx0])/(time_table[idx1]-time_table[idx0])
    else:
        return idx0, (t-time_table[idx1])/(time_table[idx1]-time_table[idx0])


if __name__ == '__main__':

    key = '00'
    rst_file = '/home/weihao/GITHUB/ORB_SLAM2-master/KeyFrameTrajectory.txt'
    # key = 'rgbd_dataset_freiburg3_nostructure_texture_near_withloop'
    # mode = 'Test'
    #key = 'rgbd_dataset_freiburg3_long_office_household'
    #mode = 'Train'

    print(key)

    # location = '/home/weihao/Projects/cambridge/OldHospital'
    # pose_file = 'dataset_train.txt'
    # poses_dic, cam = load_cambridge_poses(location, pose_file)

    if key.startswith('0'):
        location = '{}/datasets/kitti'.format(HOME)
        poses_dic, cam = load_kitti_poses(location, key)
    elif key.startswith('rgbd'):
        location = '{}/datasets/TUM'.format(HOME)
        poses_dic, cam = load_TUM_poses(location, key)
    else:
        location = "{}/datasets/indoors/{}".format(HOME, key)  # office" #heads
        poses_dic, cam = load_indoor_7_poses(location, "{}Split.txt".format(mode))

    for id in poses_dic:
        time_table_file = location + '/sequences/' + id + '/times.txt'
        time_table = np.loadtxt(time_table_file)
        poses = poses_dic[id]
        print(len(poses), len(time_table))
        rst = np.loadtxt(rst_file)

        sz = rst.shape
        for a in range(sz[0]-1):
            ar = rst[a, :]
            t = ar[0]
            a1 = [ar[7], ar[4], ar[5], ar[6]]
            a1 = Utils.q_to_m(a1)[:3, :3]
            tdx,dt = find_time(t, time_table)
            m1 = poses[tdx].m3x3

            ar = rst[a+1, :]
            t = ar[0]
            a2 = [ar[7], ar[4], ar[5], ar[6]]
            a2 = Utils.q_to_m(a2)[:3, :3]
            tdx, dt = find_time(t, time_table)
            m2 = poses[tdx].m3x3
            m = Utils.get_A(m1.dot(m2.transpose()))
            a1 = Utils.get_A(a1.dot(a2.transpose()))

            print(a, t, tdx, a1, m)


