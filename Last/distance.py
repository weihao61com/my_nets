import sys
# import tensorflow as tf
import datetime
import pickle
import os
import numpy as np
import random
import logging
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("distance")

sys.path.append('..')
from utils import Utils, Config, HOME


def distance_cal(dist, n=3, verbose=False):
    th = 1000000.
    mm = 0
    st = 1
    length = len(dist)
    dist = np.array(dist)
    while True:
        dist = dist[dist<th]
        mm = np.mean(dist)
        st = np.std(dist)
        if verbose:
            logger.info("mm, st: {} {} {}".format(len(dist), mm, st))
        th = mm + st*n
        if th>=np.max(dist):
            break
    length1 = len(dist)
    logger.info("TH/mean/fraction {} {} {}".format(th, mm, float(length1)/length))
    return th


if __name__ == '__main__':
    test_file = "c:\\tmp\\distance.p"

    if len(sys.argv)>1:
        test_file = sys.argv[1]

    logger.info("File: {}".format(test_file))

    with open(test_file, 'rb') as fp:
        dist = pickle.load(fp)
    logger.info("Data: {}".format(len(dist)))
    dist = np.array(dist)

    th = distance_cal(dist, 5, True)
    dist = dist[dist<th]

    h1 = np.histogram(dist, bins=100)
    plt.plot(h1[1][:-1], np.log(h1[0]+1), 'r-')
    plt.show()