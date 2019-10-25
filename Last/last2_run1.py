import sys
import tensorflow as tf
import datetime
import pickle
import os
import numpy as np
import random
import logging

#logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
#logger = logging.getLogger("last2")
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger("last2")

# logging.basicConfig(format='%(asctime)s - %(name)s ')
# logger = logging.getLogger("last2")
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
#logger.setFormatter(formatter)

sys.path.append('..')
sys.path.append('.')
from utils import Utils, Config, HOME
from network import Network
from Last.distance import distance_cal
from Last.last2 import create_net, DataSet2

def run(cfg, t):

    cfg.num_output = [2, 1]
    print("LR {} num_out {} mode {}".format(cfg.lr, cfg.num_output, cfg.mode))

    net = create_net(cfg)
    init = net[0]
    saver = net[1]
    input_dic = net[2]
    outputs = net[3]
    # losses = net[4]
    # opts = net[5]
    xyz = net[6]

    avg_file = Utils.avg_file_name(cfg.netFile)

    if t=='tr':
        tr = DataSet2(cfg.tr_data[0], cfg)
    else:
        tr = DataSet2(cfg.te_data[0], cfg)

    tr.subtract_avg(avg_file)
    T0 = datetime.datetime.now()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver.restore(sess, cfg.netFile)

        tr_pre = tr.prepare(None, clear=True, rd=False)
        te_count = 0
        diff_loss1 = 0
        diff_loss2 = 0

        data = tr_pre[0]
        truth = tr_pre[1]
        length = data.shape[0]
        print('data', length)
        batch_size = 10000
        r = 2000.0/length
        fp = open('/home/weihao/tmp/xyz1_{}.csv'.format(t), 'w')
        fp1 = open('/home/weihao/tmp/dist_{}.csv'.format(t), 'w')

        for c in range(0, length, batch_size):
            dd = data[c:c + batch_size]
            th = []
            for d in truth[c:c + batch_size]:
                th.append(d[2])
            te_count += len(th)

            th = np.array(th)
            th1 = th[:, :2]
            th2 = th[:, 2]

            th2 = np.array(th2).reshape((len(dd), 1))
            dd = np.array(dd)
            feed = {input_dic['data_1']: dd, input_dic['data_2']: dd}
            #        outputs[0]: th1, outputs[1]: th2}
            A = sess.run(xyz, feed_dict=feed)
            diff1 = A[0] - th1
            diff_loss1 += np.sum(diff1*diff1)
            diff2 = A[1] - th2
            diff_loss2 += np.sum(diff2*diff2)
            for a in range(len(th)):
                if random.random()<r:
                    d2 = A[1][a]
                    d1 = A[0][a]
                    t1 = th1[a]
                    t2 = th2[a]
                    f = dd[a]
                    fp.write('{},{},{},{},{},{},{},{}\n'.
                             format(t1[0], d1[0], t1[1], d1[1], t2, d2, f[6], f[7]))

            A = np.concatenate((A[0], A[1]), axis=1)
            for a in range(len(th)):
                if random.random() < r:
                    t1 = truth[c + a]
                    x0 = A[a, :]
                    id1 = t1[0][0]
                    P1 = tr.poses[id1]
                    t1 = t1[2]
                    xyz1 = Utils.xyz_tran_R(x0)
                    xyz1 = Utils.transfor_T(P1, xyz1, w2c=False)
                    xyzt = np.array(t1)
                    xyzt = Utils.xyz_tran_R(xyzt)
                    depth = xyzt[2]
                    xyzt = Utils.transfor_T(P1, xyzt, w2c=False)
                    dist = np.linalg.norm(xyz1-xyzt)
                    fp1.write('{},{},{},{},{},{},{},{}\n'.
                              format(t1[0], x0[0], t1[1], x0[1], t1[2], x0[2], depth, dist))
        fp.close()

        # print(count, t_count, t_loss/t_count, te_count, te_loss/te_count)
    T = datetime.datetime.now()
    print("Err {0} {1} {2:.6f} {3:.6f}".
                format(T-T0, te_count, diff_loss1/te_count, diff_loss2/te_count))


if __name__ == '__main__':

    config_file = "config.json"

    cfg = Config(config_file)
    cfg.num_output = list(map(int, cfg.num_output.split(',')))

    t = 'te'
    if len(sys.argv)>1:
        t = sys.argv[1]

    run(cfg, t)
