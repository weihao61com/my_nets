import sys
import tensorflow as tf
import datetime
import pickle
import os
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("fc")

sys.path.append('..')
sys.path.append('.')
from utils import Utils, Config, HOME
from network import Network
from last2 import P1Net1, create_net, DataSet


def update_cloud(data, poses, sess, xyz):
    feed = {input_dic['data_0']: np.array(dd)}
    A = sess.run(xyz, feed_dict=feed)


if __name__ == '__main__':

    config_file = "config.json"

    cfg = Config(config_file)
    cfg.mode = -1

    net = create_net(cfg)
    init = net[0]
    saver = net[1]
    input_dic = net[2]
    outputs = net[3]
    losses = net[4]
    opts = net[5]
    xys = net[6]


    if hasattr(cfg, 'inter_file'):
        tr = DataSet(None,  cfg, cfg.inter_file)
    else:
        avg_file = Utils.avg_file_name(cfg.netFile)
        tr = DataSet(cfg.tr_data[0], cfg)
        tr.get_avg(avg_file)
        tr.subtract_avg(avg_file, save_im=True)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver.restore(sess, cfg.netFile)

        t00 = datetime.datetime.now()
        st1 = ''
        data = tr.data
        poses = {}
        for d in data:
            if d[0].id not in poses:
                poses[d[0].id] = np.random.rand(6)
            if d[1].id not in poses:
                poses[d[1].id] = np.random.rand(6)


        for lp in range(cfg.loop):
            update_cloud(data, poses, sess, xys[0])

            data = tr_pre[0]
            truth = tr_pre[1]
            length = data.shape[0]

            t_loss = 0
            t_count = 0
            for c in range(0, length, cfg.batch_size):
                dd = data[c:c + cfg.batch_size]
                th = []
                for d in range(c, c+dd.shape[0]):
                    t = truth[d][4]
                    th.append(t)
                feed = {input_dic['data']: np.array(dd), output: np.array(th)}
               # _ = sess.run(opt, feed_dict=feed)
                A,_ = sess.run([loss, opt], feed_dict=feed)
                t_loss += A
                t_count += len(th)

            logger.info("Err {0:.6f}".format(t_loss/t_count))
