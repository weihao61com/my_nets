import os
import shutil
import sys
from run_colmap import get_poses, copy_images, create_image_list
import subprocess


def upload(da, location, filename, hdfs_location):
    cmd = 'opus -da {} hdfs cp file://{}/{} {}/'.format(da, location, filename, hdfs_location)
    get_stdout(cmd)


def get_stdout(cmd):
    print cmd
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.readlines()


if __name__ == '__main__':
    key = 'heads'
    mode = 'Test'
    max_match_per_image = 5
    project_dir = '/Users/weihao/Projects'
    output_dir = '/Users/weihao/Downloads/{}'.format(key)

    poses_dic, cam = get_poses(project_dir, key, mode)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    image_folder = '{}/images'.format(output_dir)
    os.mkdir(image_folder)

    image_list = copy_images(poses_dic, image_folder, 1e6)

    cmd = 'cd {0}; tar -zcvf {0}/images.tar.gz images'.format(output_dir)
    lines = get_stdout(cmd)
    shutil.rmtree(image_folder)

    print 'Total images', len(image_list)

    create_image_list(output_dir, image_list, max_match_per_image*5)

    da = 'wbu2/da1'
    hdfs_location = '/user/weihao/ajob'.format(key)
    cmd = 'opus -da {} hdfs mkdir {} '.format(da, hdfs_location)
    get_stdout(cmd)

    upload(da, output_dir, 'images.tar.gz', hdfs_location)
