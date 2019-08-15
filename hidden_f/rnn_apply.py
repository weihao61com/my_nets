import sys
# from fc_dataset import *
import tensorflow as tf
import datetime
from sortedcontainers import SortedDict
import numpy as np
import os
import pickle
import random
import copy

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils, Config
from network import Network
#from fc_dataset import DataSet
from dataset_h import DataSet, reverse
from rnn import avg_file_name
from rotation_averaging.compare import compare_rotation_matrices


def run_data_0(data, inputs, sess, xy, fname, cfg):
    att = cfg.att
    rst_dic = {}
    truth_dic = {}
    for b in data:
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

    filename = '/home/weihao/tmp/{}.csv'.format(fname)
    if sys.platform == 'darwin':
        filename = '/Users/weihao/tmp/{}.csv'.format(fname)
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


def run_data_1(data, inputs, sess, xy, cfg, rst_dic, truth_dic):
    att = cfg.att
    for b in data:
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


def reshape(a):
    ss = a[0].shape
    ll = len(a)
    ss = ss + (ll,)
    output = np.zeros(ss)
    for b in range(ll):
        output[:, :, b] = a[b]
    return output


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
        nodes = list(cfg.nodes[2])
        for a in range(len(nodes)):
            ws.append(self.create_ws('feature_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        self.ws.append(ws)

        # base
        ws = []
        ins += cfg.ref_node
        nodes_in = list(cfg.nodes[0])
        for a in range(len(nodes_in)):
            ws.append(self.create_ws('base_{}'.format(a), ins, nodes_in[a]))
            ins = nodes_in[a]
        self.ws.append(ws)

        # out
        ws = []
        nodes = list(cfg.nodes[1])
        for a in range(len(nodes)):
            ws.append(self.create_ws('out_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]

        Nout = cfg.num_output - cfg.num_output1
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

            if a < cfg.feature_len/2:
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


def get_avg_file(tr, avg_file):
    av = None
    st = None
    nt = 0
    for id in tr.data:
        ds = tr.data[id]
        for d in ds[0]:
            if nt == 0:
                av = copy.copy(d)
                st = d*d
            else:
                av += d
                st += d*d
            nt += 1
    av /= nt
    st /= nt
    st = np.sqrt(st - av*av)
    print("Saving averages:", avg_file)
    for a in range(len(av)):
        print(a, av[a], st[a])

    print(avg_file)
    with open(avg_file, 'wb') as fp:
        v = (av, st)
        pickle.dump(v, fp)

    return

if __name__ == '__main__':

    config_file = "config.json"

    test = 'te'
    if len(sys.argv)>1:
        test = sys.argv[1]

    cfg = Config(config_file)

    avg_file = avg_file_name(cfg.netFile)
    if test == 'te':
        te = DataSet([cfg.te_data[0]], cfg)
    else:
        te = DataSet([cfg.tr_data[0]], cfg)
    te.avg_correction(avg_file)

    inputs = {}

    Nout = cfg.num_output - cfg.num_output1
    setattr(cfg, 'Nout', Nout)
    output = tf.compat.v1.placeholder(tf.float32, [None, Nout])

    cfg.ref_node = list(cfg.nodes[0])[-1]
    cfg.refs = np.ones((1, cfg.ref_node))
    inputs[0] = tf.compat.v1.placeholder(tf.float32, [None, cfg.ref_node])

    for a in range(cfg.feature_len):
        inputs[a+1] = tf.compat.v1.placeholder(tf.float32, [None, cfg.att])

    input_dic = {}
    for a in range(cfg.feature_len+1):
        input_dic['input_{}'.format(a)] = inputs[a]

    net = rNet(input_dic)
    net.real_setup(cfg, SIG=(cfg.SIG==1))

    xy = SortedDict()
    for a in range(cfg.feature_len):
        n = 'output_{}'.format(a)
        if n in net.layers:
            xy[a] = net.layers['output_{}'.format(a)]
    print('output', len(xy))

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    t00 = datetime.datetime.now()
    N_total = 0
    with tf.compat.v1.Session() as sess:
        sess.run(init)

        mul = 1
        if len(sys.argv) > 2:
            mul = int(sys.argv[2])
        saver.restore(sess, cfg.netFile)

        rst_dic = {}
        truth_dic = {}
        for a in range(mul):
            tr_pre_data = te.prepare(multi=-1, rd=False)
            run_data_1(tr_pre_data, input_dic, sess, xy, cfg, rst_dic, truth_dic)

        Is = te.I
        results = []
        truth = []
        fname = 'test'

        filename = '/home/weihao/tmp/{}.csv'.format(fname)
        if sys.platform == 'darwin':
            filename = '/Users/weihao/tmp/{}.csv'.format(fname)
        fp = open(filename, 'w')
        rs = []
        list_dic = {}
        RR = []
        I = []
        first_id = None
        t_scale = np.fromiter(map(float, cfg.t_scale.split(",")), dtype=np.float)
        output_dic = {}
        ths = []
        for id in rst_dic:
            dst = np.array(rst_dic[id])
            result = np.median(dst, axis=0)

            results.append(result)
            t = truth_dic[id]
            truth.append(t)

            result = result[-1]

            dr = t - result
            r = np.linalg.norm(dr)
            rs.append(r / 10)

            if id[0] - id[1] == 1:
                mm = result
                for a in range(len(t)):
                    if a not in list_dic:
                        list_dic[a] = []
                    list_dic[a].append(t[a] - mm[a])
                    if a > 0:
                        fp.write(',')
                    fp.write('{},{}'.format(t[a], mm[a]))
                fp.write(',{}\n'.format(r))
        fp.close()

        for a in list_dic:
            vals = np.array(list_dic[a])
            md = np.median(abs(vals))
            avg = np.sqrt(np.mean(vals * vals))
            print('\t Diff=1', a, md, avg)

        tr_loss, tr_median = Utils.calculate_stack_loss_avg(np.array(results), np.array(truth), 0)

        for a in range(len(tr_loss)):
            print(a, tr_loss[a], tr_median[a])
