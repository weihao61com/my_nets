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
from distance import distance_cal
from last2 import create_net, DataSet2

def run(cfg, iterations):

    cfg.num_output = [2, 1]
    print("LR {} num_out {} mode {}".format(cfg.lr, cfg.num_output, cfg.mode))

    net = create_net(cfg)
    init = net[0]
    saver = net[1]
    input_dic = net[2]
    outputs = net[3]
    losses = net[4]
    opts = net[5]
    xyz = net[6]

    avg_file = Utils.avg_file_name(cfg.netFile)

    tr = DataSet2(cfg.tr_data[0], cfg)
    if cfg.mode==0:
        tr.get_avg(avg_file)
    else:
        tr.subtract_avg(avg_file)
    T0 = datetime.datetime.now()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        if cfg.mode == 0:
            saver.save(sess, cfg.netFile)
            exit(0)

        saver.restore(sess, cfg.netFile)

        for a in range(iterations):

            for lp in range(cfg.loop):
                tr_pre = tr.prepare(None, True)
                t_loss = 0
                t_count = 0
                te_loss = 0
                te_count = 0
                diff_loss1 = 0
                diff_loss2 = 0

                data = tr_pre[0]
                truth = tr_pre[1]
                length = data.shape[0]

                for c in range(0, length, cfg.batch_size):
                    dd = data[c:c + cfg.batch_size]
                    th = []
                    for d in truth[c:c + cfg.batch_size]:
                        th.append(d[2])
                    te_count += len(th)

                    th = np.array(th)
                    th1 = th[:, :2]
                    th2 = th[:, 2]

                    th2 = np.array(th2).reshape((len(dd), 1))
                    dd = np.array(dd)
                    feed = {input_dic['data_1']: dd, input_dic['data_2']: dd,
                            outputs[0]: th1, outputs[1]: th2}
                    A, _ = sess.run([losses, opts], feed_dict=feed)
                    diff_loss1 += A[0]
                    diff_loss2 += A[1]

                # print(count, t_count, t_loss/t_count, te_count, te_loss/te_count)
            T = datetime.datetime.now()
            print("Err {0} {1} {2:.6f} {3:.6f}, {4}".
                        format(T-T0, a,  diff_loss1/te_count, diff_loss2/te_count, te_count))
            saver.save(sess, cfg.netFile)


if __name__ == '__main__':

    config_file = "config.json"

    cfg = Config(config_file)
    cfg.num_output = list(map(int, cfg.num_output.split(',')))

    iterations = 10000

    if len(sys.argv)>1:
        cfg.mode = int(sys.argv[1])

    if len(sys.argv)>2:
        iterations = int(sys.argv[2])

    run(cfg, iterations)
