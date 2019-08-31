import os
import sys

DB_NAME = 'proj.db'
IMAGE_FOLEDER = 'images'

if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects'
    GPU = 0
    EXEC = '/Users/weihao/Applications/COLMAP.app/Contents/MacOS/colmap'
else:
    EXEC = '/usr/local/bin/colmap'
    HOME = '/home/weihao/Projects'


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def run_colmap(project_path, img_dir, step=0):
    if not os.path.exists(project_path):
        os.mkdir(project_path)

    database_path = '{}/{}'.format(project_path, DB_NAME)
    image_path = '{}/{}'.format(project_path, IMAGE_FOLEDER)
    sparse_path = '{}/sparse'.format(project_path)

    if not os.path.exists(image_path):
        os.mkdir(image_path)
        cmd = 'cp {} {}'.format(img_dir, image_path)
        run_cmd(cmd)

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
    cmd += ' --input_path {}/sparse/0'.format(project_path)
    cmd += ' --output_path {}/sparse/0/sparse.ply'.format(project_path)
    cmd += ' --output_type PLY'

    run_cmd(cmd)


if __name__ == "__main__":

    project_dir = 'tmp/heads_02'
    images_dir = 'datasets/indoors/heads/seq-02/*color.png'

    project_dir = '{}/{}'.format(HOME, project_dir)
    images_dir = '{}/{}'.format(HOME, images_dir)

    run_colmap(project_dir, images_dir)
