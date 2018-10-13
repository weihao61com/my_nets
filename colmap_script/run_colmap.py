import os
import sys
from pose_ana import *

def run_cmd(cmd):
    print(cmd)
    os.system(cmd)

project_dir = '/home/weihao/Projects'
key = 'heads'  #office" #heads
mode = 'Train'
location = "{}/datasets/indoors/{}".format(project_dir, key)
pose_file = "{}Split.txt".format(mode)
poses_dic, cam = load_indoor_7_poses(location, pose_file)

GPU = 0
EXEC = '/usr/local/bin/colmap'
project_path = '/home/weihao/Projects/colmap_features/proj1'
project_name = os.path.basename(project_path)
database_path = '{}/{}.db'.format(project_path, project_name)
image_path = '{}/images'.format(project_path)

if os.path.exists(database_path):
    os.remove(database_path)

cmd = '{} feature_extractor'.format(EXEC)
cmd += ' --database_path {}'.format(database_path)
cmd += ' --image_path {}'.format(image_path)
cmd += ' --SiftExtraction.use_gpu {}'.format(GPU)

run_cmd(cmd)

cmd = '{} exhaustive_matcher'.format(EXEC)
cmd += ' --database_path {}'.format(database_path)
cmd += ' --SiftMatching.use_gpu {}'.format(GPU)

run_cmd(cmd)


