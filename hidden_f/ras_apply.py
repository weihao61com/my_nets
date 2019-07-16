import sys
# from fc_dataset import *
import tensorflow as tf
import datetime
from sortedcontainers import SortedDict
import numpy as np
import os
from dataset import DataSet
import cPickle
import cv2

HOME = '/home/weihao/Projects/'
if sys.platform == 'darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils, Config
from ras import avg_file_name
from network import Network

# from fc_dataset import DataSet

def get_depth_file(filename):
    f = filename.replace('home', 'Users', 1)
    return f.replace('color', 'depth', 1)

if __name__ == '__main__':

    config_file = "config.json"

    # if len(sys.argv) > 1:
    # config_file = sys.argv[1]

    test = 'te'
    if len(sys.argv) > 1:
        test = sys.argv[1]

    cfg = Config(config_file)

    avg_file = avg_file_name(cfg.netFile)
    rst_file = avg_file_name(cfg.netFile, 'rst')

    if test == 'te':
        tr = DataSet([cfg.te_data[0]], cfg)
    else:
        tr = DataSet([cfg.tr_data[0]], cfg)

    tr.read_rst(rst_file)
    tr.avg_apply_ras(avg_file)

    N1 = 0
    N2 = 0
    N3 = 0
    for data_id in tr.rasd.features:
        features = tr.rasd.features[data_id]
        print data_id, len(features)
        for img_id in features:
            feature = features[img_id][0]
            hidden = features[img_id][1]
            pose = tr.rasd.poses[data_id][img_id]
            depth_file = get_depth_file(pose.filename)
            depth = cv2.imread(depth_file, -1).astype(np.float16)

            #mx = np.max(depth)
            #depth0 = depth/mx*256
            #cv2.imwrite('tmp.png', depth0.astype(np.uint8))

            for feature_id in range(len(feature)):
                f = feature[feature_id]
                h = hidden[feature_id]
                if h.count() == 1:
                    N1 += 1
                    continue
                elif h.count() == 20:
                    N2 += 1
                else:
                    N3 += 1
                d = h.val()[0]
                try:
                    x = int(f[1])
                    y = int(f[0])
                    if abs(x-120)<5 and abs(y-200)<5:
                        t = depth[x, y]
                        print "rst", data_id, img_id, x, y, d, t
                except:
                    print "error", data_id, img_id, f

    print N1, N2, N3




    print
