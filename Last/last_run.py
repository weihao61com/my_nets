import sys
# from fc_dataset import *
import tensorflow as tf
import datetime
from sortedcontainers import SortedDict
import numpy as np
import os
import pickle
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("fc")

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'
elif sys.platform=='win32':
     HOME = 'c:\\Projects\\'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils, Config
from network import Network
#from fc_dataset import DataSet
from dataset import DataSet
from last import get_avg_file, rNet, run_data



def run_data_1(data, inputs, sess, xy, cfg, rst_dic):

    sz = data[0].shape
    att = 4
    for a in xy:
        if a not in rst_dic:
            rst_dic[a] = {}

    for x in range(0, sz[0], cfg.batch_size):
        feed = {}
        b = data[0][x:x + cfg.batch_size]
        t = data[1][x:x + cfg.batch_size]
        i = data[2][x:x + cfg.batch_size]
        ss = b.shape
        for a in range(int(ss[1]/att)):
            feed[input_dic['input_{}'.format(a+1)]] = b[:, att * a:att * (a + 1)]

        feed[inputs['input_0']] = np.repeat(cfg.refs, ss[0], axis=0)

        r = sess.run(xy, feed_dict=feed)
        r = np.array(r.values())

        #for a in range(o.shape[0]):
        #    dd.append(t[a, cfg.out_offset:] - opt_out[:, a])

        for a in rst_dic:
            for b in range(len(i)):
                tr = t[b, :]
                id = i[b] #[a-cfg.out_offset][0]
                rs = r[a-cfg.out_offset, b, :]
                if id not in rst_dic[a]:
                    rst_dic[a][id] = []
                rst_dic[a][id].append((rs, tr))


def run_test(input_dic, sess, xy, te, cfg, mul=1):

    rst_dic = {}
    for a in range(mul):
        tr_pre_data = te.prepare(multi=-1, rdd=False)
        run_data_1(tr_pre_data, input_dic, sess, xy, cfg, rst_dic)

    tr_loss, tr_median = run_data(rst_dic, 'test')

    for a in range(len(tr_loss)):
        print( a, tr_loss[a], tr_median[a])



if __name__ == '__main__':

    config_file = "config.json"

    test = 'te'
    if len(sys.argv)>1:
        test = sys.argv[1]

    cfg = Config(config_file)

    avg_file = Utils.avg_file_name(cfg.netFile)

    if test == 'te':
        tr = DataSet([cfg.te_data[0]], cfg, sub_sample=0.1)
    else:
        tr = DataSet([cfg.tr_data[0]], cfg)

    tr.subtract_avg(avg_file, save_im=False)

    inputs = {}
    output = {}

    cfg.Nout = 3
    cfg.att = 4

    cfg.ref_node = cfg.nodes[0][-1]
    cfg.refs = np.ones(cfg.ref_node)
    cfg.refs = cfg.refs.reshape((1, cfg.ref_node))
    inputs[0] = tf.placeholder(tf.float32, [None, cfg.ref_node])

    for a in range(cfg.feature_len):
        inputs[a+1] = tf.placeholder(tf.float32, [None, cfg.att])

    input_dic = {}
    for a in range(cfg.feature_len+1):
        input_dic['input_{}'.format(a)] = inputs[a]

    net = rNet(input_dic)
    net.real_setup(cfg, SIG=(cfg.SIG==1))

    xy = SortedDict()

    Nout = cfg.Nout
    #if cfg.fc_Nout>0:
    #    Nout = cfg.fc_Nout*2
    for a in range(cfg.feature_len):
        n = 'output_{}'.format(a)
        if n in net.layers:
            xy[a] = net.layers['output_{}'.format(a)]
            output[a] = tf.placeholder(tf.float32, [None, Nout])
    print('output count:', len(xy))
    #
    # loss = None
    # last_loss = None
    # for a in xy:
    #     # print (a)
    #     if cfg.L1==0:
    #         last_loss = tf.reduce_sum(tf.square(tf.subtract(xy[a], output[a])))
    #     else:
    #         last_loss = tf.reduce_sum(tf.abs(tf.subtract(xy[a], output[a])))
    #     if loss is None:
    #         loss = last_loss
    #     else:
    #         loss = loss + last_loss
    #
    # opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9,
    #                 beta2=0.999, epsilon=0.00000001,
    #                 use_locking=False, name='Adam').\
    #     minimize(loss)
    # # opt = tf.train.GradientDescentOptimizer(learning_rate=cfg.lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    t00 = datetime.datetime.now()
    N_total = 0
    with tf.Session() as sess:
        sess.run(init)

        mul = 1
        if len(sys.argv) > 2:
            mul = int(sys.argv[2])
        saver.restore(sess, cfg.netFile)
        run_test(input_dic, sess, xy, tr, cfg, mul)
