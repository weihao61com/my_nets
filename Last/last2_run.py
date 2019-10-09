import sys
import tensorflow as tf
import datetime
import pickle
import os
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("last2_run")

sys.path.append('..')
sys.path.append('.')
from utils import Utils, Config, HOME
from network import Network
from last2 import P1Net1, create_net, DataSet2

F1 = 0.1
F2 = 1-F1

def compare_truth(data):

    err = 0
    nt = 0
    for d in data:
        for id in range(2):
            dd = d[id]
            tran = dd.Q4[:3, 3]
            distance = np.linalg.norm(tran-dd.tran)
            err += distance*distance
            nt += 1
    logger.warn("Compare Error: {} {}".format(np.sqrt(err/nt), nt))

def set_pose(data, Q):
    data.Q4 = Q
    data.inv = np.linalg.inv(Q)
    # data.m3x3 = Q[:3, :3]
    # data.tran = Q[:3, 3]


def run_data(n, xyz, data,  input_dic, sess):
    pr = DataSet2.prepare_data(data, n)
    inputs = []
    for p in pr:
        inputs.append(p[0])
    inputs = np.array(inputs)

    feed = {input_dic['data_{}'.format(n)]: inputs}
    A = sess.run(xyz, feed_dict=feed)
    return A


def update_A(data, input_dic, sess, xyz, n):
    A0 = run_data(n, xyz, data, input_dic, sess)
    logger.warn("update A {} {}".format(n, A0.shape))

    cloud = {}
    nt = 0
    for d in data:
        id1 = d[0].id
        id2 = d[1].id
        if id1 not in cloud:
            cloud[id1] = []
        if id2 not in cloud:
            cloud[id2] = []

        length = len(d[2])
        for b in range(length):
            A, T = Utils.get_relative(d[0], d[1])
            A[n-3] = A0[nt]
            Q = d[0].Q4.dot(Utils.create_Q(A, T))
            A = Utils.get_A(Q)
            cloud[id2].append(A)
            nt += 1

            A, T = Utils.get_relative(d[1], d[0])
            A[n-3] = A0[nt]
            Q = d[1].Q4.dot(Utils.create_Q(A, T))
            A = Utils.get_A(Q)
            cloud[id1].append(A)
            nt += 1

    err = 0
    nt = 0
    for id in cloud:
        d = np.array(cloud[id])
        st = np.std(d, axis=0)
        nt += d.shape[0]
        err += np.sum(st*st)
        cloud[id] = np.mean(d, axis=0)
    logger.warn('A error {} {} {}'.format(np.sqrt(n, err/nt), nt))

    for d in data:
        id1 = d[0].id
        id2 = d[1].id
        A, T = Utils.get_A_T(d[0].Q4)
        A = cloud[id1]
        d[0].Q4 = Utils.create_Q(A, T)
        d[0].inv = np.linalg.inv(d[0].Q4)
        A, T = Utils.get_A_T(d[1].Q4)
        A = cloud[id2]
        d[1].Q4 = Utils.create_Q(A, T)
        d[1].inv = np.linalg.inv(d[1].Q4)


def update_T(data, input_dic, sess, xyz):
    A = run_data(2, xyz, data, input_dic, sess)
    logger.warn("update T {}".format(A.shape))

    cloud = {}
    nt = 0
    for d in data:
        id1 = d[0].id
        id2 = d[1].id
        if id1 not in cloud:
            cloud[id1] = []
        if id2 not in cloud:
            cloud[id2] = []

        length = len(d[2])
        for b in range(length):
            xyz = transfor_T(d[0], A[nt, :], w2c=False)
            cloud[id2].append(xyz)
            nt += 1
            xyz = transfor_T(d[1], A[nt, :], w2c=False)
            cloud[id1].append(xyz)
            nt += 1

    err = 0
    nt = 0
    for id in cloud:
        d = np.array(cloud[id])
        st = np.std(d, axis=0)
        nt += d.shape[0]
        err += np.sum(st*st)
        cloud[id] = np.mean(d, axis=0)
    logger.warn('T error {} {}'.format(np.sqrt(err/nt), nt))

    for d in data:
        id1 = d[0].id
        id2 = d[1].id
        A, T0 = Utils.get_A_T(d[0].Q4)
        T = cloud[id1]
        T = T*F1+T0*F2
        d[0].Q4 = Utils.create_Q(A, T)
        d[0].inv = np.linalg.inv(d[0].Q4)
        A, T0 = Utils.get_A_T(d[1].Q4)
        T = cloud[id2]
        T = T*F1+T0*F2
        d[1].Q4 = Utils.create_Q(A, T)
        d[1].inv = np.linalg.inv(d[1].Q4)



def update_C(data, input_dic, sess, xyz):
    A = run_data(1, xyz, data, input_dic, sess)
    logger.warn("update cloud {}".format(A.shape))

    cloud = {}
    nt = 0
    for d in data:
        id1 = d[0].id
        id2 = d[1].id
        if id1 not in cloud:
            cloud[id1] = {}
        if id2 not in cloud:
            cloud[id2] = {}
        length = len(d[2])
        for b in range(length):
            c = d[2][b]
            p_id1 = int(c[3])
            p_id2 = int(c[4])
            if p_id1 not in cloud[id1]:
                cloud[id1][p_id1] = []
            if p_id2 not in cloud[id2]:
                cloud[id2][p_id2] = []

            xyz = transfor_T(d[0], A[nt, :], w2c=False)
            cloud[id1][p_id1].append(xyz)
            cloud[id2][p_id2].append(xyz)
            nt += 1

            xyz = transfor_T(d[1], A[nt, :], w2c=False)
            cloud[id1][p_id1].append(xyz)
            cloud[id2][p_id2].append(xyz)
            nt += 1

    err = 0
    nt = 0
    for id in cloud:
        for p_id in cloud[id]:
            d = np.array(cloud[id][p_id])
            st = np.std(d, axis=0)
            nt += d.shape[0]
            err += np.sum(st*st)
            cloud[id][p_id] = np.mean(d, axis=0)
    logger.warn('Cloud error {} {}'.format(np.sqrt(err/nt), nt))

    nt = 0
    err = 0
    for d in data:
        length = len(d[2])
        id1 = d[0].id
        id2 = d[1].id
        for b in range(length):
            c = d[2][b]
            p_id1 = int(c[3])
            p_id2 = int(c[4])
            xyz1 = cloud[id1][p_id1]
            xyz2 = cloud[id2][p_id2]
            xyz = (xyz1 + xyz2)/2
            xyz0 = d[2][b][0]
            err += np.linalg.norm(xyz1-xyz2)
            nt += 1
            m1 = d[2][b][1]
            m2 = d[2][b][2]
            xyz = xyz*F1+xyz0*F2
            d[2][b] = xyz, m1, m2, p_id1, p_id2
    logger.warn('Cloud distance {} {}'.format(err / nt, nt))

if __name__ == '__main__':

    config_file = "config.json"

    cfg = Config(config_file)
    cfg.mode = -1
    cfg.num_output = list(map(int, cfg.num_output.split(',')))

    net = create_net(cfg)
    init = net[0]
    saver = net[1]
    input_dic = net[2]
    outputs = net[3]
    losses = net[4]
    opts = net[5]
    xys = net[6]

    avg_file = Utils.avg_file_name(cfg.netFile)
    tr = DataSet2(cfg.tr_data[0], cfg)
    Utils.create_cloud(tr)
    tr.get_avg(avg_file)
    tr.subtract_avg(avg_file)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # saver.restore(sess, cfg.netFile)

        t00 = datetime.datetime.now()
        st1 = ''


        for lp in range(cfg.loop):
            update_C(data, input_dic, sess, xys[0])
            update_T(data, input_dic, sess, xys[1])
            update_A(data, input_dic, sess, xys[2], 3)
            update_A(data, input_dic, sess, xys[3], 4)
            update_A(data, input_dic, sess, xys[4], 5)
            
            compare_truth(data)
            #with open(os.path.join(HOME, cfg.t_out), 'wb') as fp:
            #    pickle.dump(data, fp)
