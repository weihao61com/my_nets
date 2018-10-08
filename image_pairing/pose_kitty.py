from sortedcontainers import SortedDict
import numpy as np
from utils import Pose, PinholeCamera, rotationMatrixToEulerAngles


def load_kitty_poses(location, pose_file):
    poses = SortedDict()
    filename = '{}/poses/{}.txt'.format(location, pose_file)
    id = 0
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            poses[id] = Pose(line, location, pose_file)
            id += 1

    return poses


location = '/Users/weihao/BlueNoteData/dataset'
pose_file = '00'
poses = load_kitty_poses(location, pose_file)

focal = 719 #719
# cam = PinholeCamera(1241.0, 376.0, focal, focal, 607.1928, 185.2157)
cam = PinholeCamera(1241.0, 376.0, focal, focal, 620.5, 188.0)

print('Total images {}'.format(len(poses)))

with open('w.txt', 'w') as fp:
    pose_pre = None
    for id in poses:
        pose = poses[id]
        if pose_pre is not None:
            t = pose.get_tran(pose_pre)
            d = rotationMatrixToEulerAngles(pose.get_direction(pose_pre, cam))
            fp.write('{} {} {} {} {} {} {}\n'.format(id, d[0], d[1], d[2], t[0],t[1], t[2]))
        pose_pre = pose