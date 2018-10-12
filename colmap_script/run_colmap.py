import os


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


EXEC = '/usr/local/bin/colmap'
project_path = '/home/weihao/Projects/colmap_features/proj2'
project_name = os.path.basename(project_path)

database_path = '{}/{}.db'.format(project_path, project_name)
image_path = '{}/images'.format(project_path)

cmd = '{} feature_extractor --database_path {} --image_path {}'.format(EXEC, database_path, image_path)

run_cmd(cmd)

cmd = '{} exhaustive_matcher --database_path {}'.format(EXEC, database_path)

run_cmd(cmd)


