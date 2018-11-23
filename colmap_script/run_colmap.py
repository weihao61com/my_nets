import os
import sys
import shutil
import random

this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/../image_pairing'.format(this_file_path))
from pose_ana import load_indoor_7_poses

DB_NAME = 'proj.db'
IMAGE_FOLEDER = 'images'
MATCHES_LIST = 'matches.txt'

def copy_images(poses_dic, image_path, MAX_image):
    image_list = []
    if os.path.exists(image_path):
        shutil.rmtree(image_path)
    os.mkdir(image_path)

    nt = 0
    for pid in poses_dic:
        poses = poses_dic[pid]
        for id in poses:
            pose = poses[id]
            basename = '{}_{}'.format(pid, os.path.basename(pose.filename))
            image_name = image_path + '/{}'.format( basename)
            shutil.copy(pose.filename, image_name)
            image_list.append(basename)
            nt += 1
            if nt>=MAX_image:
                break
        if nt>=MAX_image:
            break
    return image_list


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)

def get_seq(filename):
    strs = filename.split('.')
    strs = strs[0].split('-')
    id = int(strs[1])
    seq = int(strs[0].split('_')[0])
    return id, seq

def create_image_seq_list(database_path, image_list, seq = 3):

    image_pairs = {}
    for img1 in image_list:
        id1, seq1 = get_seq(img1)
        for img2 in image_list:
            id2, seq2 = get_seq(img2)
            if seq1==seq2 and 0<abs(id1-id2)<=seq:
                if img1 not in image_pairs:
                    image_pairs[img1] = []
                image_pairs[img1].append(img2)

    total = 0
    matches_list = os.path.join(database_path, MATCHES_LIST)
    with open(matches_list, 'w') as fp:
        for img in image_pairs:
            for ig in image_pairs[img]:
                fp.write('{} {}\n'.format(img, ig))
                total += 1
    print 'Total pair', total

def create_image_list(database_path, image_list, pair_count = 100):
    if pair_count>len(image_list):
        raise Exception('More pair than imges')
    image_pairs = {}
    indexes = range(len(image_list))
    for img in image_list:
        random.shuffle(indexes)
        image_pairs[img] = []
        a = 0
        while pair_count>len(image_pairs[img]):
            idx = indexes[a]
            if image_list[idx] != img:
                image_pairs[img].append(image_list[idx])
            a += 1
    # dup
    total_removed = 0
    for img in image_pairs:
        removed = []
        for ip in image_pairs[img]:
            if img in image_pairs[ip]:
                removed.append(ip)
                break
        if len(removed)>0:
            for r in removed:
                image_pairs[img].remove(r)
                total_removed += 1
    print 'removed dup', total_removed

    total = 0
    matches_list = os.path.join(database_path, MATCHES_LIST)
    with open(matches_list, 'w') as fp:
        for img in image_pairs:
            for ig in image_pairs[img]:
                fp.write('{} {}\n'.format(img, ig))
                total += 1
    print 'Total pair', total


def get_poses(project_dir, key, mode):
    location = "{}/datasets/indoors/{}".format(project_dir, key)
    pose_file = "{}Split.txt".format(mode)
    poses_dic, cam = load_indoor_7_poses(location, pose_file)
    return  poses_dic, cam

def run_colmap(project_dir, key, mode, max_image, max_match_per_image):

    #location = "{}/datasets/indoors/{}".format(project_dir, key)
    #pose_file = "{}Split.txt".format(mode)
    #poses_dic, cam = load_indoor_7_poses(location, pose_file)

    poses_dic, cam = get_poses(project_dir, key, mode)

    GPU = 0
    EXEC = '/usr/local/bin/colmap'
    project_path = '{}/colmap_features/{}_{}'.format(project_dir,key, mode)
    # project_name = os.path.basename(project_path)
    database_path = '{}/{}'.format(project_path, DB_NAME)
    image_path = '{}/{}'.format(project_path, IMAGE_FOLEDER)

    if not os.path.exists(project_path):
        os.mkdir(project_path)

    if os.path.exists(database_path):
        os.remove(database_path)

    image_list = copy_images(poses_dic, image_path, max_image)

    cmd = '{} feature_extractor'.format(EXEC)
    cmd += ' --database_path {}'.format(database_path)
    cmd += ' --image_path {}'.format(image_path)
    cmd += ' --SiftExtraction.use_gpu {}'.format(GPU)
    cmd += ' --SiftExtraction.num_threads 6'
    cmd += ' --ImageReader.default_focal_length_factor 0.82'
    cmd += ' --ImageReader.single_camera 1'

    run_cmd(cmd)

    if max_match_per_image<0:
        create_image_seq_list(project_path, image_list, -max_match_per_image)
    else:
        create_image_list(project_path, image_list, max_match_per_image*5)

    cmd = '{} matches_importer'.format(EXEC)
    cmd += ' --database_path {}'.format(database_path)
    cmd += ' --SiftMatching.use_gpu {}'.format(GPU)
    cmd += ' --SiftMatching.num_threads 6'
    cmd += ' --match_list_path {}'.format(os.path.join(project_path, MATCHES_LIST))

    run_cmd(cmd)

    # shutil.rmtree(image_path)

if __name__ == "__main__":
    key = 'heads'  #office" #heads
    mode = 'Test'
    project_dir = '/home/weihao/Projects'

    run_colmap(project_dir, key, mode, 2000)