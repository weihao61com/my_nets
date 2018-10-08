# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
import cv2
import glob
import sys


if __name__ == '__main__':
    import os
    config_file = "config7.json"
    rep = None

    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if len(sys.argv) > 2:
        rep = int(sys.argv[2])
        cmd = 'python train.py {} {}'.format(config_file, rep)
        os.system(cmd)
        cmd = 'python test.py {} {}'.format(config_file, rep)
        os.system(cmd)
        cmd = 'python test.py {} {} -1'.format(config_file, rep)
        os.system(cmd)
    else:
        for rep in range(-1, 7):
            cmd = 'python train.py {} {}'.format(config_file, rep)
            print cmd
            os.system(cmd)


        for rep in range(-1, 7):
            cmd = 'python test.py {} {}'.format(config_file, rep)
            os.system(cmd)
            cmd = 'python test.py {} {} -1'.format(config_file, rep)
            os.system(cmd)