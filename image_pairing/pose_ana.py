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

def load_indoor_7_poses(location, pose_file):
    poses = SortedDict()
    filename = '{}/{}'.format(location, pose_file)
    id = 0
    with open(filename) as f:
        for line in f.readlines():
            print line, len(line)
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

def load_kitty_poses(location, pose_file):
    poses = SortedDict()
    filename = '{}/poses/{}.txt'.format(location, pose_file)
    id = 0
    with open(filename) as f:
        for line in f.readlines():
            poses[id] = Pose(line, location, pose_file, data=0, id=id)
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
                seq = strs[0][:4]
                if not seq in poses:
                    poses[seq] = SortedDict()
                id = int(strs[0][10:15])
                p = Pose(line, location, pose_file, data=1)
                if os.path.exists(p.filename):
                    poses[seq][id-1] = Pose(line, location, pose_file, data=1)
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


if __name__ == '__main__':

    location = '/Users/weihao/BlueNoteData/cambridge/ShopFacade'
    pose_file = '/dataset_train.txt'

    poses = get_pose(location, pose_file)
    with open('w.txt', 'w') as fp:
        for id in poses:
            p = poses[id]
            fp.write('{} {} {} {} {} {} {}\n'.format(id, p[0], p[1], p[2], p[3], p[4], p[5]))


