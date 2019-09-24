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


def run_data_0(b, inputs, sess, xy, fname, cfg):
    att = cfg.att
    rst_dic = {}
    truth_dic = {}
    length = int(b[0].shape[1]/att)
    feed = {}
    b_sz = b[0].shape[0]

    feed[inputs['input_0']] = np.repeat(cfg.refs, b_sz,  axis=0)
    for a in range(length):
        feed[inputs['input_{}'.format(a+1)]] = b[0][:, att * a:att * (a + 1)]
    result = []
    for a in xy:
        r = sess.run(xy[a], feed_dict=feed)
        result.append(r)

    result = np.array(result)
    for a in range(len(b[2])):
        if not b[2][a] in rst_dic:
            rst_dic[b[2][a]] = []
        rst_dic[b[2][a]].append(result[:, a, :])
        truth_dic[b[2][a]] = b[1][a]

    results = []
    truth = []

    filename = '{}/../tmp/{}.csv'.format(HOME, fname)
    fp = open(filename, 'w')
    rs = []
    for id in rst_dic:
        dst = np.array(rst_dic[id])
        result = np.median(dst, axis=0)
        results.append(result)
        truth.append(truth_dic[id])
        t = truth_dic[id]
        r = np.linalg.norm(t - result[-1])
        rs.append(r*r)
        if random.random() < 0.2:
            mm = result[-1]
            for a in range(len(t)):
                if a > 0:
                    fp.write(',')
                fp.write('{},{}'.format(t[a], mm[a]))
            fp.write(',{}\n'.format(r))
    fp.close()
    rs = sorted(rs)
    length = len(rs)
    fp = open(filename+'.csv', 'w')
    for a in range(length):
        fp.write('{},{}\n'.format(float(a)/length, rs[a]))
    fp.close()

    return Utils.calculate_stack_loss_avg(np.array(results), np.array(truth), cfg.L1)


def calculate_rst(rs):
    s = []
    for r in rs:
        s.append(r[0])
    s = np.array(s)
    return rs[0][1], np.mean(s, 0)


def run_data(rst_dic, fname):
    filename = '{}/tmp/{}.csv'.format(HOME, fname)
    a_last = list(rst_dic.keys())[-1]

    M = []
    L = []
    for a in rst_dic:
        results = []
        truth = []
        rst = rst_dic[a]
        fp = None
        if a==a_last:
            fp = open(filename, 'w')

        for id in rst:
            rt = rst_dic[a][id]
            t, r = calculate_rst(rt)
            results.append(r)
            truth.append(t)
            r0 = np.linalg.norm(r-t)
            if fp is not None:
                fp.write('{},{},{},{},{},{},{}\n'.
                         format(r[0], t[0], r[1], t[1], r[2], t[2], r0))

        m, l = Utils.calculate_stack_loss_ras\
            (np.array(results), np.array(truth), 0)
        M.append(m)
        L.append(l)

    return M, L

class rNet(Network):

    def create_ws(self, n, ins, outs):
        print(n, ins, outs)
        w = self.make_var('weights_{}'.format(n), shape=[ins, outs])
        b = self.make_var('biases_{}'.format(n), shape=[outs])
        return [w,b]

    def parameters(self, cfg):

        self.ws = []

        # feature
        ws = []
        ins = cfg.att
        nodes = cfg.nodes[2]
        for a in range(len(nodes)):
            ws.append(self.create_ws('feature_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        self.ws.append(ws)

        # base
        ws = []
        ins += cfg.ref_node
        nodes_in = cfg.nodes[0]
        for a in range(len(nodes_in)):
            ws.append(self.create_ws('base_{}'.format(a), ins, nodes_in[a]))
            ins = nodes_in[a]
        self.ws.append(ws)

        # out
        ws = []
        nodes = cfg.nodes[1]
        for a in range(len(nodes)):
            ws.append(self.create_ws('out_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]

        Nout = cfg.Nout
        ws.append(self.create_ws('out', ins, Nout))
        self.ws.append(ws)

    def setup(self):
        pass

    def real_setup(self, cfg, SIG=False):
        self.parameters(cfg)

        ref_out = 'input_0'
        for a in range(0, cfg.feature_len):
            inputs = 'input_{}'.format(a+1)
            self.feed(inputs)
            for b in range(len(self.ws[0])):
                n = 'input_{}_{}'.format(a, b)
                if SIG:
                    self.fc_ws(ws=self.ws[0][b], name=n)
                else:
                    self.fc_w2(ws=self.ws[0][b], name=n)
                f_out = n

            self.feed(f_out, ref_out).concat(1, name='f_inputs_{}'.format(a))
            for b in range(len(self.ws[1])):
                n = 'base_{}_{}'.format(a, b)
                if SIG:
                    self.fc_ws(ws=self.ws[1][b], name=n)
                else:
                    self.fc_w2(ws=self.ws[1][b], name=n)
                ref_out = n

            if a < cfg.out_offset:
                continue

            self.feed(ref_out)
            for b in range(len(self.ws[2])):
                if b < len(self.ws[2])-1:
                    n = 'output_{}_{}'.format(a, b)
                    if SIG:
                        self.fc_ws(ws=self.ws[2][b], name=n)
                    else:
                        self.fc_w2(ws=self.ws[2][b], name=n)
                else:
                    n = 'output_{}'.format(a)
                    if SIG:
                        self.fc_ws(ws=self.ws[2][b], name=n, sig=False)
                    else:
                        self.fc_w2(ws=self.ws[2][b], name=n, relu=False)


def get_avg_file(tr, filename):
        x = []
        for d in tr.data:
            for a in d[2]:
                x.append(a[1])
                x.append(a[2])

        x = np.array(x)
        avx = np.mean(x, axis=0)
        stx = np.std(x, axis=0)

        print(avg_file)
        with open(filename, 'wb') as fp:
            pickle.dump((avx, stx), fp)


if __name__ == '__main__':

    config_file = "config.json"

    cfg = Config(config_file)
    avg_file = Utils.avg_file_name(cfg.netFile)

    if hasattr(cfg, 'inter_file'):
        tr = DataSet(None,  cfg, cfg.inter_file)
    else:
        tr = DataSet(cfg.tr_data[0], cfg)
        tr.get_avg(avg_file)
        tr.subtract_avg(avg_file, save_im=True)


    tr.subtract_avg(avg_file, save_im=False)
    iterations = 1000000
    loop = cfg.loop
    logger.info("LR {} num_out {} mode {}".format(cfg.lr, cfg.num_output, cfg.mode))

    inputs = {}
    output = {}
    lr = cfg.lr
    learning_rate = tf.placeholder(tf.float32, shape=[])

    #cfg.Nout = 3
    #cfg.att = 4

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

    loss = None
    last_loss = None
    for a in xy:
        # print (a)
        if cfg.L1==0:
            last_loss = tf.reduce_sum(tf.square(tf.subtract(xy[a], output[a])))
        else:
            last_loss = tf.reduce_sum(tf.abs(tf.subtract(xy[a], output[a])))
        if loss is None:
            loss = last_loss
        else:
            loss = loss + last_loss

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9,
                    beta2=0.999, epsilon=0.00000001,
                    use_locking=False, name='Adam').\
        minimize(loss)
    # opt = tf.train.GradientDescentOptimizer(learning_rate=cfg.lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    t00 = datetime.datetime.now()
    N_total = 0
    with tf.Session() as sess:
        sess.run(init)

        if test is not None:
            mul = 1
            if len(sys.argv) > 2:
                mul = int(sys.argv[2])
            saver.restore(sess, cfg.netFile)
            run_test(input_dic, sess, xy, tr, cfg, mul)

        if cfg.renetFile:
            logger.info("Re net: {}".format(cfg.renetFile))
            saver.restore(sess, cfg.renetFile)

        str1 = ''

        te_data = te.prepare()
        tr_data = tr0.prepare()

        for a in range(iterations):

            t1 = datetime.datetime.now()
            str = "it: {0:.3f} {1:.3f} {2:4.2e}".\
                format(a*loop/1000.0, (t1 - t00).total_seconds()/3600.0, lr)

            tr_loss, tr_median = run_data_0(tr_data, input_dic, sess, xy, 'tr', cfg)
            te_loss, te_median = run_data_0(te_data, input_dic, sess, xy, 'te', cfg)

            s = -1
            while True:
                s += int(len(tr_loss)/2)
                str += " {0:.3f} {1:.3f} {2:.3f} {3:.3f} ".format(tr_loss[s], te_loss[s], tr_median[s], te_median[s])
                if s==len(tr_loss)-1:
                    break

            print(str, str1)

            if lr<1e-9:
                break

            tl3 = 0
            to3 = 0
            r3 = 0
            nt = 0
            att = cfg.att
            for _ in range(loop):
                tr_pre_data = tr.prepare(multi=5)
                while tr_pre_data:
                    sz = tr_pre_data[0].shape
                    length = sz[1]

                    for x in range(0, sz[0], cfg.batch_size):
                        feed = {learning_rate: lr}
                        b = tr_pre_data[0][x:x+cfg.batch_size]
                        o = tr_pre_data[1][x:x+cfg.batch_size]
                        ids = tr_pre_data[2][x:x+cfg.batch_size]

                        n0 = b.shape[0]
                        for a in range(int(length/att)):
                            feed[input_dic['input_{}'.format(a+1)]] = b[:, att * a:att * (a + 1)]
                            if a in output:
                                feed[output[a]] = o
                        feed[input_dic['input_0']] = np.repeat(cfg.refs, n0, axis=0)

                        # opt_out = sess.run(xy, feed_dict=feed)
                        ll3,_ = sess.run([loss, opt], feed_dict=feed)

                        tl3 += ll3
                        nt += n0
                    tr_pre_data = None #tr.get_next(avg=avg_file)
            N_total += 1
            if N_total % cfg.INC_win == 0:
                lr -= cfg.d_lr

            tl3 /= nt
            to3 /= nt
            r3 /= nt
            str1 = "{0:.4f}".format(tl3)
            Utils.save_tf_data(saver, sess, cfg.netFile)


