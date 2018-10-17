import os

DB_NAME = 'proj.db'
IMAGE_FOLEDER = 'images'
GPU = 0
EXEC = '/usr/local/bin/colmap'


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def run_colmap(project_path):
    database_path = '{}/{}'.format(project_path, DB_NAME)
    image_path = '{}/{}'.format(project_path, IMAGE_FOLEDER)

    cmd = '{} feature_extractor'.format(EXEC)
    cmd += ' --database_path {}'.format(database_path)
    cmd += ' --image_path {}'.format(image_path)
    cmd += ' --SiftExtraction.use_gpu {}'.format(GPU)
    cmd += ' --SiftExtraction.num_threads 6'
    cmd += ' --ImageReader.single_camera 1'
    cmd += ' --ImageReader.default_focal_length_factor 0.82'


    run_cmd(cmd)

    # cmd = '{} matches_importer'.format(EXEC)
    cmd = '{} exhaustive_matcher'.format(EXEC)
    cmd += ' --database_path {}'.format(database_path)
    cmd += ' --SiftMatching.use_gpu {}'.format(GPU)
    cmd += ' --SiftMatching.num_threads 1'
    #cmd += ' --match_list_path {}'.format(os.path.join(project_path, MATCHES_LIST))

    run_cmd(cmd)


if __name__ == "__main__":

    project_dir = '/home/weihao/tmp/office'

    run_colmap(project_dir)
