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

# model = RandomForestRegressor()
# model.fit(train_X,train_y)
#
# # Get the mean absolute error on the validation data
# predicted_prices = model.predict(val_X)
# MAE = mean_absolute_error(val_y , predicted_prices)
# print('Random forest validation MAE = ', MAE)

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

    locs = ['L_0', 'L_1', 'L_2', 'L_3', 'L_4', 'L_5', 'L_6', 'L_7', 'L_8', 'L_9']
    dd = l_utils.get_dataset('/home/weihao/Projects/p_files/L10000', locs)

    CV = 5


    for c in range(CV):
        truth, features = l_utils.prepare_data(dd[0], -c-1)
        print len(truth)
        rf = RandomForestRegressor()
        #rf = RN()

        rf.fit(features, truth)
        truth, features = l_utils.prepare_data(dd[0], c+1)
        print len(truth)
        rst1 = rf.predict(features)
        print 'error1', np.mean(np.abs(np.array(rst1)-np.array(truth)))
        for a in range(0, len(truth), 10):
            print rst1[a], truth[a]

        break
