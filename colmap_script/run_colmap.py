import os


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


GPU = 0
EXEC = '/usr/local/bin/colmap'
project_path = '/home/weihao/Projects/colmap_features/proj1'
project_name = os.path.basename(project_path)
database_path = '{}/{}.db'.format(project_path, project_name)
image_path = '{}/images'.format(project_path)

cmd = '{} feature_extractor'.format(EXEC)
cmd += ' --database_path {}'.format(database_path)
cmd += ' --image_path {}'.format(image_path)
cmd += ' --SiftExtraction.use_gpu {}'.format(GPU)

run_cmd(cmd)

cmd = '{} exhaustive_matcher'.format(EXEC)
cmd += ' --database_path {}'.format(database_path)
cmd += ' --SiftMatching.use_gpu {}'.format(GPU)

run_cmd(cmd)


