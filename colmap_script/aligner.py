import sys
import numpy as np
import os
sys.path.append('..')

from utils import HOME
from image_pairing.pose_ana import \
    load_kitti_poses, load_indoor_7_poses, load_TUM_poses


if __name__ == '__main__':
    range2 = 1
    range3 = -range2

    read_time = False
    key = 'heads'
    mode = 'Test'
    # key = 'rgbd_dataset_freiburg3_nostructure_texture_near_withloop'
    # mode = 'Test'
    #key = 'rgbd_dataset_freiburg3_long_office_household'
    #mode = 'Train'

    if len(sys.argv)>1:
        key = sys.argv[1]
    if len(sys.argv)>2:
        mode = sys.argv[2]
    if len(sys.argv)>3:
        read_time = True

    if key.startswith('0'):
        location = '{}/datasets/kitti'.format(HOME)
        poses_dic, cam = load_kitti_poses(location, key)
        key = 'kitti_{}'.format(key)
    elif key.startswith('rgbd'):
        location = '{}/datasets/TUM'.format(HOME)
        poses_dic, cam = load_TUM_poses(location, key)
    else:
        location = "{}/datasets/indoors/{}".format(HOME, key)  # office" #heads
        poses_dic, cam = load_indoor_7_poses(location, "{}Split.txt".format(mode))

    if read_time:
        for id in poses_dic:
            time_table_file = location + '/sequences/' + id + '/times.txt'
            time_table = np.loadtxt(time_table_file)
            poses = poses_dic[id]
            print(len(poses), len(time_table))

    print(key, mode, range2, range3)

    for id in poses_dic:
        for pose in poses_dic[id]:
            filename = os.path.basename(poses_dic[id][pose].filename)
            tran = poses_dic[id][pose].tran
            #print(filename, tran[0], tran[1], tran[2])

    EXEC = '/Users/weihao/Applications/COLMAP.app/Contents/MacOS/colmap'
    EXEC += ' model_aligner'
    EXEC += ' --input_path {}/tmp/{}/sparse/0/'.format(HOME, key)
    EXEC += ' --output_path {}/tmp/{}/align/'.format(HOME, key)
    EXEC += ' --robust_alignment 1'
    EXEC += ' --robust_alignment_max_error 0.01'
    EXEC += ' --ref_images_path {}/tmp/aligner.txt'.format(HOME)
    print(EXEC)
    #  => Alignment error: 0.014240 (mean), 0.012230 (median)
