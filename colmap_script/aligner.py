import sys
import numpy as np
import os
sys.path.append('..')

from utils import HOME, Utils
from image_pairing.pose_ana import \
    load_kitti_poses, load_indoor_7_poses, load_TUM_poses

def create_aligner(poses_dic, project_dir, images_dir):

    if len(poses_dic)!=1:
        raise Exception("Single run please")

    filename = os.path.join(project_dir, 'aligner.txt')
    with open(filename, 'w') as fp:
        nt = 0
        for id in poses_dic:
            poses = poses_dic[id]
            for pose_id in poses:
                pose = poses[pose_id]
                image_name = pose.filename
                cmd = 'cp {} {}'.format(image_name, images_dir)
                Utils.run_cmd(cmd)
                tran = pose.tran
                fp.write('{} {} {} {}\n'.
                         format(os.path.basename(image_name), tran[0], tran[1], tran[2]))
                nt += 1


if __name__ == '__main__':

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

    create_aligner(key, mode, 'aligner.txt')

    EXEC = '/Users/weihao/Applications/COLMAP.app/Contents/MacOS/colmap'
    EXEC += ' model_aligner'
    EXEC += ' --input_path {}/tmp/{}/sparse/0/'.format(HOME, key)
    EXEC += ' --output_path {}/tmp/{}/align/'.format(HOME, key)
    EXEC += ' --robust_alignment 1'
    EXEC += ' --robust_alignment_max_error 0.01'
    EXEC += ' --ref_images_path {}/tmp/aligner.txt'.format(HOME)
    print(EXEC)
    #  => Alignment error: 0.014240 (mean), 0.012230 (median)
