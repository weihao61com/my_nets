import random
import glob
import os

MIN_num=1000
MAX_num =10000

location = '/home/weihao/Downloads/redkitchen'
zs = glob.glob(os.path.join(location, '*.zip'))
for z in zs:
    cmd = 'unzip {} -d {}'.format(z, location)
    print cmd
    os.system(cmd)