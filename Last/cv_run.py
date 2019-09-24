import sys
# from fc_dataset import *
# import tensorflow as tf
import datetime
from sortedcontainers import SortedDict
import numpy as np
import os
import pickle
import random
import logging
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("cv_run")

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'
elif sys.platform=='win32':
     HOME = 'c:\\Projects\\'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils, Config
# from network import Network
#from fc_dataset import DataSet
from dataset import DataSet

filename = '{}/p_files/office_Test_cv_s6_2.p'.format(HOME)


def camera(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def cv_process(data):
    focal = 525.0
    w2 = 320
    h2 = 240
    mx = camera(focal, focal, w2, h2)

    print(len(data))
    rs = []
    angs = []
    rs0 = []
    rs1 = []
    rs2 = []
    att = 0

    output_file = '{}/tmp/feature.csv'.format(HOME)
    fp = open(output_file, 'w')

    for d in data:

        P1 = d[0]
        P2 = d[1]
        A, T = Utils.get_relative(P1, P2)
        d0 = d[2]
        if len(d0)<10:
            continue

        px_new = []
        px_last = []
        for c in d0:
            px_new.append(c[2])
            px_last.append(c[1])

        px_new = np.array(px_new)
        px_last = np.array(px_last)
        # print(px_new.shape)

        E, mask = cv2.findEssentialMat(px_new, px_last, cameraMatrix=mx, method=cv2.RANSAC)
        mh, R, t, mask0 = cv2.recoverPose(E, px_new, px_last, cameraMatrix=mx)
        #(E)
        #print(R)
        #print(t)
        b = Utils.get_A(R)
        e = []
        for c in b:
            # if c<-90:
            #     c = 180 + c
            # if c>90:
            #     c = 180-c
            e.append(c)
        b = e
        a = A
        T0 = np.linalg.norm(T)
        a = T/T0
        b = t[:, 0]

        dr = a - b
        r0 = np.linalg.norm(dr)
        #if r0>180:
        #    r0 = r0 - 180
        #elif r0>90:
        #    r0 = 180 - r0
        rs.append(r0*r0)
        rs0.append(abs(dr[0]))
        rs1.append(abs(dr[1]))
        rs2.append(abs(dr[2]))
        angs.append(np.linalg.norm(a))
        if len(rs)%200 == 0:
            logger.info("{}: {}".format(len(rs), len(data)))

        if random.random() > 200.0/len(data):
            continue

        fp.write('{},{},{},{},{},{},{},{},{},{},{}\n'.
                 format(a[0], a[1], a[2], b[0], b[1], b[2], r0, mh, len(d0), np.sum(mask),T0))
    print('att', att/len(data))
    angs = np.array(angs)
    rs0 = np.array(rs0)
    rs1 = np.array(rs1)
    rs2 = np.array(rs2)

    #rs = np.sqrt(rs)
    print ('name, median, Anger-error, mx, my, mz')
    print ('{0}, {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}'.format(
        os.path.basename(filename), np.sqrt(np.median(rs)), np.median(angs),
        np.median(rs0),np.median(rs1),np.median(rs2)))
    print ('Average(RMS), {0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}'.format(
        np.sqrt(np.mean(rs)), np.sqrt(np.mean(angs*angs)),
        np.sqrt(np.mean(rs0*rs0)),np.sqrt(np.mean(rs1*rs1)),
        np.sqrt(np.mean(rs2*rs2))))
    #print '{0}, {1:.4f} '.format(
    #    os.path.basename(filename), np.mean(np.sqrt(rs)))
    fp.close()

    rs = sorted(rs)
    length = len(rs)
    fp = open(output_file + '.csv', 'w')
    for a in range(length):
        fp.write('{},{}\n'.format(rs[a], float(a) / length))
    fp.close()


if __name__ == '__main__':

    config_file = "config.json"

    test = 'te'
    cfg = Config(config_file)

    if len(sys.argv) > 1:
        test = sys.argv[1]

    if test == 'te':
        te_file = cfg.te_data
    else:
        te_file = cfg.tr_data
    logger.info("test {} ".format(te_file))

    te = DataSet(te_file, cfg)
    # te.data = te.data[:20]
    cv_process(te.data)
