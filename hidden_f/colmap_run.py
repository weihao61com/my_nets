import os
import sys
import shutil

this_file_path = os.path.dirname(os.path.realpath(__file__))
HOME = '{}/../..'.format(this_file_path)
sys.path.append('{}/my_nets'.format(HOME))

from image_pairing.pose_ana import \
    load_kitti_poses, load_indoor_7_poses, load_TUM_poses
from utils import Utils


def run_cmd(cmd, verbose=False):
    if verbose:
        print cmd
    os.system(cmd)


if __name__ == '__main__':

    COLMAP = '/usr/local/bin/colmap'
    if sys.platform == 'darwin':
        COLMAP = '/Users/weihao/Applications/COLMAP.app/Contents/MacOS/colmap'

    key = '00'
    mode = 'Train'

    COLMAP = '/usr/local/bin/colmap'
    wd = '/home/weihao/tmp/{}'.format(key)
    if sys.platform == 'darwin':
        COLMAP = '/Users/weihao/Applications/COLMAP.app/Contents/MacOS/colmap'
        wd = '/Users/weihao/tmp/{}'.format(key)

    if key.startswith('0'):
        location = '{}/datasets/kitti'.format(HOME)
        poses_dic, cam = load_kitti_poses(location, key + ".txt")
        key = 'kitti_{}'.format(key)
    elif key.startswith('rgbd'):
        location = '{}/datasets/TUM'.format(HOME)
        poses_dic, cam = load_TUM_poses(location, key)
    else:
        location = "{}/datasets/indoors/{}".format(HOME, key)  # office" #heads
        poses_dic, cam = load_indoor_7_poses(location, "{}Split.txt".format(mode))

    stage = 0

    if stage < 1:
        if os.path.exists(wd):
            shutil.rmtree(wd)
        os.mkdir(wd)

        image_dir = '{}/images'.format(wd)
        os.mkdir(image_dir)
        for id in poses_dic:
            poses = poses_dic[id]
            for p_id in poses:
                pose = poses[p_id]
                cmd = 'cp {} {}'.format(pose.filename, image_dir)
                run_cmd(cmd)

    if stage < 2:
        cmd = '{0} feature_extractor --database_path {1}/database.db --image_path {1}/images'.format(COLMAP, wd)
        run_cmd(cmd, True)

    if stage < 3:
        cmd = '{0} exhaustive_matcher --database_path {1}/database.db'.format(COLMAP, wd)
        run_cmd(cmd, True)

    if stage < 4:
        cmd = 'mkdir {}/sparse'.format(wd)
        run_cmd(cmd, True)

        cmd = '{0} mapper --database_path {1}/database.db ' \
              '--image_path {1}/images --output_path {1}/sparse'.format(COLMAP, wd)
        run_cmd(cmd, True)

    if stage < 5:
        cmd = '{0} model_converter --input_path {1}/sparse/0 ' \
              '--output_path {1}/sparse/0/point.ply --output_type PLY'.format(COLMAP, wd)
        run_cmd(cmd, True)

    #
    # $ mkdir $DATASET_PATH / dense
    #
    # $ colmap
    # image_undistorter \
    # - -image_path $DATASET_PATH / images \
    #                - -input_path $DATASET_PATH / sparse / 0 \
    #                               - -output_path $DATASET_PATH / dense \
    #                                               - -output_type
    # COLMAP \
    # - -max_image_size
    # 2000
    #
    # $ colmap
    # patch_match_stereo \
    # - -workspace_path $DATASET_PATH / dense \
    #                    - -workspace_format
    # COLMAP \
    # - -PatchMatchStereo.geom_consistency
    # true
    #
    # $ colmap
    # stereo_fusion \
    # - -workspace_path $DATASET_PATH / dense \
    #                    - -workspace_format
    # COLMAP \
    # - -input_type
    # geometric \
    # - -output_path $DATASET_PATH / dense / fused.ply
    #
    # $ colmap
    # poisson_mesher \
    # - -input_path $DATASET_PATH / dense / fused.ply \
    #                - -output_path $DATASET_PATH / dense / meshed - poisson.ply
    #
    # $ colmap
    # delaunay_mesher \
    # - -input_path $DATASET_PATH / dense \
    #                - -output_path $DATASET_PATH / dense / meshed - delaunay.ply