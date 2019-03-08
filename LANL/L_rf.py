import sys
import tensorflow as tf
import datetime
import numpy as np
import glob
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from rt import RN

sys.path.append('..')
from network import Network
from utils import Utils


def get_dataset(st='../../p_files/L_*.p', avg_file='../../p_files/Avg.p'):

    files = glob.glob(st)
    data = []
    for f in files:
        with open(f, 'r') as fp:
            d = pickle.load(fp)
            data.append(d)

    if not os.path.exists(avg_file):
        d = None
        for dd in data:
            if d is None:
                d = dd[1]
            else:
                d = np.concatenate((d, dd[1]))
        avg = np.mean(d, 0)
        std = np.std(d, 0)
        with open(avg_file, 'w') as fp:
            pickle.dump((avg, std), fp)
    else:
        with open(avg_file, 'r') as fp:
            A = pickle.load(fp)
        avg = A[0]
        std = A[1]

    for n in range(len(data)):
        for a in range(data[n][1].shape[0]):
            data[n][1][a, :] = (data[n][1][a,:]-avg)/std

    return data


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


def prepare_data(data):
    t = None
    d = None
    for a in data:
        if t is None:
            t = a[0]
            d = a[1]
        else:
            t = np.concatenate((t, a[0]))
            d = np.concatenate((d, a[1]))
    return t, d


if __name__ == '__main__':

    data = get_dataset()
    CV = len(data)
    # nodes = [124, 32]
    # lr = 1e-4
    # iterations = 1000
    # loop = 10
    # batch_size = 100
    # netFile = '../../NNs/L_{}'

    for c in range(CV):
        te, tr = create_data(data, c)
        # rf = RandomForestRegressor(1000)
        rf = RN()

        t, d = prepare_data(tr)
        rf.fit(d, t)
        t, d = prepare_data(te)

        rst = rf.predict(d)
        print 'error', np.mean(np.abs(rst-t))
