import sys
import tensorflow as tf
import datetime
import numpy as np
import glob
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from rt import RN
from LANL_Utils import l_utils

sys.path.append('..')
from network import Network
from utils import Utils


def create_data(data, id):
    tr = []
    te = None
    nt = 0
    for d in data:
        if nt == id:
            te = [d]
        else:
            tr.append(d)

        nt += 1

    return te, tr


def run_data(data, inputs, sess, xy):
    results = None
    truth = None

    for b in data:
        feed = {inputs: b[1]}
        result = sess.run(xy, feed_dict=feed)[:, 0]
        if results is None:
            results = result
            truth = b[0]
        else:
            results = np.concatenate((results, result))
            truth = np.concatenate((truth, b[0]))
    return np.mean(np.abs(results-truth))


if __name__ == '__main__':

    data = l_utils.get_dataset('/home/weihao/Projects/p_files/L', 'L1_*.p')
    CV = len(data)
    # nodes = [124, 32]
    # lr = 1e-4
    # iterations = 1000
    # loop = 10
    # batch_size = 100
    # netFile = '../../NNs/L_{}'

    for c in range(CV):
        te, tr = create_data(data, c)
        rf = RandomForestRegressor(1000)
        #rf = RN()

        d, t0 = l_utils.prepare_data(tr)
        rf.fit(d, t0)
        rst0 = rf.predict(d)

        d, t1 = l_utils.prepare_data(te)
        rst1 = rf.predict(d)
        print 'error0', np.mean(np.abs(np.array(rst0)-np.array(t0)))
        print 'error1', np.mean(np.abs(np.array(rst1)-np.array(t1)))
        #for a in range(len(t1)):
        #    print rst1[a], t1[a]
        #for a in range(len(t0)):
        #    print rst0[a], t0[a]
        #break