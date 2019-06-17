from sortedcontainers import SortedDict
import numpy as np
from imagery_utils import Pose
import os
import sys

this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/..'.format(this_file_path))
from utils import Utils, PinholeCamera
from glob import glob
import os


def load_TUM_poses(location, pose_file):
    file_dir = os.path.join(location, pose_file)
    rgb_file = os.path.join(file_dir, 'rgb.txt')
    rgb = SortedDict()
    T0 = None
    with open(rgb_file) as f:
        for line in f.readlines()[3:]:
            strs = line.split(' ')
            t = float(strs[0])
            if T0 is None:
                T0 = t
            t -= T0
            rgb[t] = strs[1][:-1]

    ps = SortedDict()
    rgb_file = os.path.join(file_dir, 'groundtruth.txt')
    with open(rgb_file) as f:
        for line in f.readlines()[3:]:
            strs = map(float, line[:-1].split(' '))
            t = strs[0] - T0
            ps[t] = np.array(strs[1:])

    id = 0
    poses = SortedDict()
    for t in rgb:
        p = Pose(file_dir, interp_poses(ps, t), rgb[t], data=1)
        poses[id] = p
        id += 1

    focal = 525.0
    cam = PinholeCamera(640.0, 480.0, focal, focal, 320.0, 240.0)
    # cam = PinholeCamera(480.0, 640.0, focal, focal, 240.0, 320.0)

    return {pose_file: poses}, cam


def load_indoor_7_poses(location, pose_file):
    poses = SortedDict()
    filename = '{}/{}'.format(location, pose_file)
    id = 0
    with open(filename) as f:
        for line in f.readlines():
            print line[:-2], len(line)
            if len(line)>8:
                folder_id = int(line[8:-2])
            else:
                folder_id = int(line[4:-2])
            if not folder_id in poses:
                poses[folder_id] = SortedDict()

            folder_name = "seq-{0:02d}".format(folder_id)

            files = glob(os.path.join(location, folder_name, '*.color.png'))
            for f in files:
                p = Pose(line, location, f, data=2)
                poses[folder_id][p.id] = p

    focal = 525.0
    cam = PinholeCamera(640.0, 480.0, focal, focal, 320.0, 240.0)
    # cam = PinholeCamera(480.0, 640.0, focal, focal, 240.0, 320.0)

    return poses, cam

def load_kitti_poses(location, pose_file):
    poses = SortedDict()
    filename = '{}/poses/{}'.format(location, pose_file)
    id = 0
    with open(filename) as f:
        for line in f.readlines():
            poses[id] = Pose(line, location, pose_file[:-4], data=0, id=id)
            id += 1

    focal = 719  # 719
    cam = PinholeCamera(1241.0, 376.0, focal, focal, 607.1928, 185.2157)

    return {pose_file: poses}, cam


def load_cambridge_poses(location, pose_file):
    import os
    poses = SortedDict()
    filename = '{}/{}'.format(location, pose_file)
    cnt = -3
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            if cnt>=0:
                strs = line.split(" ")
                seq = strs[0].split('/')
                if not seq[0] in poses:
                    poses[seq[0]] = SortedDict()
                id = int(seq[1][5:10])
                p = Pose(line, location, strs[0], data=1)
                if os.path.exists(p.filename):
                    poses[seq[0]][id] = p
                else:
                    raise Exception('file not found: {}'.format(p.filename))

            cnt += 1

    focal = 1400*0.7*.7
    cam = PinholeCamera(1920.0, 1080.0, focal, focal, 960, 540)

    return poses, cam


def get_pose(location, pose_file):
    poses = load_cambridge_poses(location, pose_file)

    out_pose = SortedDict()
    focal = 719 #719
    # cam = PinholeCamera(1241.0, 376.0, focal, focal, 607.1928, 185.2157)
    cam = PinholeCamera(1241.0, 376.0, focal, focal, 620.5, 188.0)

    print('Total images {}'.format(len(poses)))

    pose_pre = None
    for id in poses:
        pose = poses[id]
        if pose_pre is not None:
            t = pose.get_tran(pose_pre)
            d = Utils.rotationMatrixToEulerAngles(pose.get_direction(pose_pre, cam))
            out_pose[id] = [d[0], d[1], d[2], t[0], t[1], t[2]]
            # fp.write('{} {} {} {} {} {} {}\n'.format(id, d[0], d[1], d[2], t[0], t[1], t[2]))
        pose_pre = pose

    return {'0': out_pose}


def interp_poses(ps, tp):
    keys = ps.keys()
    tl = keys[0]
    th = keys[-1]
    for t in keys[1:]:
        if t>tp:
            th = t
            break
        tl = t

    pl = ps[tl]
    ph = ps[th]
    dp = ph-pl
    dt = th-tl
    p = pl + dp*(tp-tl)/(dt)
    return p


if __name__ == '__main__':

    #location = '/home/weihao/Projects/datasets/cambridge/StMarysChurch'
    #pose_file = 'dataset_train.txt'
    location = '/home/weihao/Projects/datasets/indoors/office'
    pose_file = 'TestSplit.txt'
    # location = '/home/weihao/Projects/datasets/kitti'
    # pose_file = '00.txt'

    poses = None
    if 'indoor' in location:
        poses, _ = load_indoor_7_poses(location, pose_file)
    if 'kitti' in location:
        poses, _ = load_kitti_poses(location, pose_file)
    if 'cambridge' in location:
        poses, _ = load_cambridge_poses(location, pose_file)

    if poses is None:
        raise Exception('Poses not found {} {}'.format(location, pose_file))
    with open('/home/weihao/tmp/w.csv', 'w') as fp:
        for id in poses:
            ps = poses[id]
            print id, len(ps)
            pre_pose = None
            for p_id in ps:
                p = ps[p_id]
                if pre_pose is not None:
                    Q1 = pre_pose.Q4
                    Q2 = p.Q4
                    fp.write('{},{}'.format(id,p_id))
                    A, T = Utils.get_A_T(Q1)
                    fp.write(',{},{},{},{},{},{}'.
                             format(A[0], A[1], A[2], T[0], T[1], T[2]))
                    A, T = Utils.get_relative_A_T(Q1, Q2)
                    fp.write(',{},{},{},{},{},{}\n'.
                             format(A[0], A[1], A[2], T[0], T[1], T[2]))
                pre_pose = p
            break
