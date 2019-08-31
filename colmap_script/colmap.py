import os
import sys

sys.path.append('..')

from utils import HOME
from image_pairing.pose_ana import \
    load_kitti_poses, load_indoor_7_poses, load_TUM_poses

from aligner import create_aligner

DB_NAME = 'proj.db'
IMAGE_FOLEDER = 'images'
GPU = 0

if sys.platform == 'darwin':
    HOME = '/Users/weihao/Projects'
    EXEC = '/Users/weihao/Applications/COLMAP.app/Contents/MacOS/colmap'
else:
    EXEC = '/usr/local/bin/colmap'
    HOME = '/home/weihao/Projects'


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def run_colmap(project_path, img_dir, step=0):

    database_path = '{}/{}'.format(project_path, DB_NAME)
    image_path = img_dir
    sparse_path = '{}/sparse'.format(project_path)

    if not os.path.exists(sparse_path):
        os.mkdir(sparse_path)

    cmd = '{} feature_extractor'.format(EXEC)
    cmd += ' --database_path {}'.format(database_path)
    cmd += ' --image_path {}'.format(image_path)
    cmd += ' --SiftExtraction.use_gpu {}'.format(GPU)
    cmd += ' --SiftExtraction.num_threads 6'
    cmd += ' --ImageReader.single_camera 1'
    # cmd += ' --ImageReader.default_focal_length_factor 0.82'

    run_cmd(cmd)

    #cmd = '{} exhaustive_matcher'.format(EXEC)
    #cmd += ' --database_path {}'.format(database_path)
    #cmd += ' --SiftMatching.use_gpu {}'.format(GPU)

    cmd = '{} sequential_matcher'.format(EXEC)
    cmd += ' --database_path {}'.format(database_path)
    cmd += ' --SiftMatching.use_gpu {}'.format(GPU)
    cmd += ' --SiftMatching.num_threads 6'
    cmd += ' --SequentialMatching.overlap 10'
    cmd += ' --SequentialMatching.loop_detection 1'

    run_cmd(cmd)

    cmd = '{} mapper'.format(EXEC)
    cmd += ' --database_path {}'.format(database_path)
    cmd += ' --Mapper.ba_global_use_pba {}'.format(GPU)
    cmd += ' --image_path {}'.format(image_path)
    cmd += ' --output_path {}'.format(sparse_path)

    run_cmd(cmd)

    cmd = '{} model_converter'.format(EXEC)
    cmd += ' --input_path {}/sparse/0/'.format(project_path)
    cmd += ' --output_path {}/sparse/0/sparse.ply'.format(project_path)
    cmd += ' --output_type PLY'

    run_cmd(cmd)

    os.mkdir('{}/sparse/align_0'.format(project_path))

    cmd = '{} model_aligner'.format(EXEC)
    cmd += ' --input_path {}/sparse/0'.format(project_path)
    cmd += ' --output_path {}/sparse/align_0/'.format(project_path)
    cmd += ' --robust_alignment 1'
    cmd += ' --robust_alignment_max_error 0.01'
    cmd += ' --ref_images_path {}/aligner.txt'.format(project_path)
    cmd += ' > {}/sparse/align_0/rst.txt'.format(project_path)
    run_cmd(cmd)

    cmd = '{} model_converter'.format(EXEC)
    cmd += ' --input_path {}/sparse/align_0/'.format(project_path)
    cmd += ' --output_path {}/sparse/align_0/sparse.ply'.format(project_path)
    cmd += ' --output_type PLY'

    run_cmd(cmd)


if __name__ == "__main__":

    key = 'heads'
    mode = 'Test'
    # key = 'rgbd_dataset_freiburg3_nostructure_texture_near_withloop'
    # mode = 'Test'
    # key = 'rgbd_dataset_freiburg3_long_office_household'
    # mode = 'Train'

    if len(sys.argv) > 1:
        key = sys.argv[1]
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    if len(sys.argv) > 3:
        read_time = True

    print(key, mode)

    # location = '/home/weihao/Projects/cambridge/OldHospital'
    # pose_file = 'dataset_train.txt'
    # poses_dic, cam = load_cambridge_poses(location, pose_file)

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


    project_dir = 'tmp/{}_{}'.format(key, mode)
    images_dir = '{}/images'.format(project_dir)
    project_dir = '{}/{}'.format(HOME, project_dir)
    images_dir = '{}/{}'.format(HOME, images_dir)

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    create_aligner(poses_dic, project_dir, images_dir)

    run_colmap(project_dir, images_dir)
