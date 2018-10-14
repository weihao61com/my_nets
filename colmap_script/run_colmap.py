import os
import sys
import shutil

this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/../image_pairing'.format(this_file_path))
from pose_ana import load_indoor_7_poses
MAX_image = 200

def copy_images(poses_dic, image_path):
    if os.path.exists(image_path):
        shutil.rmtree(image_path)
    os.mkdir(image_path)

    nt = 0
    for pid in poses_dic:
        poses = poses_dic[pid]
        for id in poses:
            pose = poses[id]
            basename = os.path.basename(pose.filename)
            shutil.copy(pose.filename, image_path + '/{}_{}'.format(pid, basename))
            nt += 1
            if nt>=MAX_image:
                break
        if nt>=MAX_image:
            break


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def get_poses(project_dir, key, mode):
    location = "{}/datasets/indoors/{}".format(project_dir, key)
    pose_file = "{}Split.txt".format(mode)
    poses_dic, cam = load_indoor_7_poses(location, pose_file)
    return  poses_dic, cam

def run_colmap(project_dir, key, mode):

    #location = "{}/datasets/indoors/{}".format(project_dir, key)
    #pose_file = "{}Split.txt".format(mode)
    #poses_dic, cam = load_indoor_7_poses(location, pose_file)

    poses_dic, cam = get_poses(project_dir, key, mode)

    GPU = 0
    EXEC = '/usr/local/bin/colmap'
    project_path = '{}/colmap_features/{}_{}'.format(project_dir,key, mode)
    project_name = os.path.basename(project_path)
    database_path = '{}/proj.db'.format(project_path)
    image_path = '{}/images'.format(project_path)

    if not os.path.exists(project_path):
        os.mkdir(project_path)

    if os.path.exists(database_path):
        os.remove(database_path)

    copy_images(poses_dic, image_path)

    cmd = '{} feature_extractor'.format(EXEC)
    cmd += ' --database_path {}'.format(database_path)
    cmd += ' --image_path {}'.format(image_path)
    cmd += ' --SiftExtraction.use_gpu {}'.format(GPU)
    cmd += ' --SiftExtraction.num_threads 6'
    cmd += ' --ImageReader.default_focal_length_factor 0.82'
    cmd += ' --ImageReader.single_camera 1'

    run_cmd(cmd)

    cmd = '{} exhaustive_matcher'.format(EXEC)
    cmd += ' --database_path {}'.format(database_path)
    cmd += ' --SiftMatching.use_gpu {}'.format(GPU)
    cmd += ' --SiftMatching.num_threads 6'

    run_cmd(cmd)

    # shutil.rmtree(image_path)

if __name__ == "__main__":
    key = 'heads'  #office" #heads
    mode = 'Test'
    project_dir = '/home/weihao/Projects'

    run_colmap(project_dir, key, mode)