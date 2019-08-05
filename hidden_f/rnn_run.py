import sys
# from fc_dataset import *
import tensorflow as tf
import datetime
from sortedcontainers import SortedDict
import numpy as np
import os
import random

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils, Config
from network import Network
from dataset_h import DataSet
from rnn import avg_file_name, rNet, run_data_1

#
# def run_data_0(data, inputs, sess, xy, fname, cfg):
#     att = cfg.att
#     rst_dic = {}
#     truth_dic = {}
#     for b in data:
#         length = b[0].shape[1]/att
#         feed = {}
#         b_sz = b[0].shape[0]
#
#         feed[inputs['input_0']] = np.repeat(cfg.refs, b_sz,  axis=0)
#         for a in range(length):
#             feed[inputs['input_{}'.format(a+1)]] = b[0][:, att * a:att * (a + 1)]
#         result = []
#         for a in xy:
#             r = sess.run(xy[a], feed_dict=feed)
#             result.append(r)
#
#         result = np.array(result)
#         for a in range(len(b[2])):
#             if not b[2][a] in rst_dic:
#                 rst_dic[b[2][a]] = []
#             rst_dic[b[2][a]].append(result[:, a, :])
#             truth_dic[b[2][a]] = b[1][a]
#
#     results = []
#     truth = []
#
#     filename = '/home/weihao/tmp/{}.csv'.format(fname)
#     if sys.platform == 'darwin':
#         filename = '/Users/weihao/tmp/{}.csv'.format(fname)
#     fp = open(filename, 'w')
#     rs = []
#     for id in rst_dic:
#         dst = np.array(rst_dic[id])
#         result = np.median(dst, axis=0)
#         results.append(result)
#         truth.append(truth_dic[id])
#         t = truth_dic[id]
#         r = np.linalg.norm(t - result[-1])
#         rs.append(r*r)
#         if random.random() < 0.2:
#             mm = result[-1]
#             for a in range(len(t)):
#                 if a > 0:
#                     fp.write(',')
#                 fp.write('{},{}'.format(t[a], mm[a]))
#             fp.write(',{}\n'.format(r))
#     fp.close()
#     rs = sorted(rs)
#     length = len(rs)
#     fp = open(filename+'.csv', 'w')
#     for a in range(length):
#         fp.write('{},{}\n'.format(float(a)/length, rs[a]))
#     fp.close()
#
#     return Utils.calculate_stack_loss_avg(np.array(results), np.array(truth), cfg.L1)
#
#
# def run_data_1(data, inputs, sess, xy, cfg, rst_dic, truth_dic, imgs_dic):
#     att = cfg.att
#     for b in data:
#         length = int(b[0].shape[1]/att)
#         feed = {}
#         b_sz = b[0].shape[0]
#
#         feed[inputs['input_0']] = np.repeat(cfg.refs, b_sz,  axis=0)
#         for a in range(length):
#             feed[inputs['input_{}'.format(a+1)]] = b[0][:, att * a:att * (a + 1)]
#         result = []
#         for a in xy:
#             r = sess.run(xy[a], feed_dict=feed)
#             result.append(r)
#
#         result = np.array(result)
#         for a in range(len(b[2])):
#             if not b[3][a] in rst_dic:
#                 rst_dic[b[3][a]] = []
#             rst_dic[b[3][a]].append(result[:, a, :])
#             truth_dic[b[3][a]] = b[1][a]
#             # imgs_dic[b[3][a]] = b[2][a]
#
#
# def run_data(rst_dic, truth_dic, imgs_dic, fname):
#
#     results = []
#     truth = []
#
#     filename = '/home/weihao/tmp/{}.csv'.format(fname)
#     if sys.platform == 'darwin':
#         filename = '/Users/weihao/tmp/{}.csv'.format(fname)
#     fp = open(filename, 'w')
#     rs = []
#     for id in rst_dic:
#         dst = np.array(rst_dic[id])
#         result = np.median(dst, axis=0)
#         # result = np.mean(dst, axis=0)
#         print(dst.shape, result.shape)
#         results.append(result)
#         truth.append(truth_dic[id])
#         t = truth_dic[id]
#         dr = t - result[-1]
#         r = np.linalg.norm(dr)
#         rs.append(r*r)
#
#         if random.random() < 1.2:
#             mm = result[-1]
#             for a in range(len(t)):
#                 if a>0:
#                     fp.write(',')
#                 fp.write('{},{}'.format(t[a], mm[a]))
#             fp.write(',{}\n'.format(r))
#             # if len(mm)==3:
#             #     fp.write('{},{},{},{},{},{},{}\n'.
#             #          format(t[0], mm[0], t[1], mm[1], t[2], mm[2], r))
#             # else:
#             #     fp.write('{},{},{}\n'.
#             #              format(t, mm[0], r))
#     fp.close()
#     rs = sorted(rs)
#     length = len(rs)
#     fp = open(filename+'.csv', 'w')
#     for a in range(length):
#         fp.write('{},{}\n'.format(float(a)/length, rs[a]))
#     fp.close()
#
#     v1 = np.array(results)
#     v1 = v1[:, -1, :]
#     return v1
#
# class rNet(Network):
#
#     def create_ws(self, n, ins, outs):
#         print(n, ins, outs)
#         w = self.make_var('weights_{}'.format(n), shape=[ins, outs])
#         b = self.make_var('biases_{}'.format(n), shape=[outs])
#         return [w,b]
#
#     def parameters(self, cfg):
#
#         self.ws = []
#
#         # feature
#         ws = []
#         ins = cfg.att
#         nodes = cfg.nodes[2]
#         for a in range(len(nodes)):
#             ws.append(self.create_ws('feature_{}'.format(a), ins, nodes[a]))
#             ins = nodes[a]
#         self.ws.append(ws)
#
#         # base
#         ws = []
#         ins += cfg.ref_node
#         nodes_in = cfg.nodes[0]
#         for a in range(len(nodes_in)):
#             ws.append(self.create_ws('base_{}'.format(a), ins, nodes_in[a]))
#             ins = nodes_in[a]
#         self.ws.append(ws)
#
#         # out
#         ws = []
#         nodes = cfg.nodes[1]
#         for a in range(len(nodes)):
#             ws.append(self.create_ws('out_{}'.format(a), ins, nodes[a]))
#             ins = nodes[a]
#
#         Nout = cfg.num_output - cfg.num_output1
#         ws.append(self.create_ws('out', ins, Nout))
#         self.ws.append(ws)
#
#     def setup(self):
#         pass
#
#     def real_setup(self, cfg, SIG=False):
#         self.parameters(cfg)
#
#         ref_out = 'input_0'
#         for a in range(0, cfg.feature_len):
#             inputs = 'input_{}'.format(a+1)
#             self.feed(inputs)
#             for b in range(len(self.ws[0])):
#                 n = 'input_{}_{}'.format(a, b)
#                 if SIG:
#                     self.fc_ws(ws=self.ws[0][b], name=n)
#                 else:
#                     self.fc_w2(ws=self.ws[0][b], name=n)
#                 f_out = n
#
#             self.feed(f_out, ref_out).concat(1, name='f_inputs_{}'.format(a))
#             for b in range(len(self.ws[1])):
#                 n = 'base_{}_{}'.format(a, b)
#                 if SIG:
#                     self.fc_ws(ws=self.ws[1][b], name=n)
#                 else:
#                     self.fc_w2(ws=self.ws[1][b], name=n)
#                 ref_out = n
#
#             if a < cfg.feature_len/2:
#                 continue
#
#             self.feed(ref_out)
#             for b in range(len(self.ws[2])):
#                 if b < len(self.ws[2])-1:
#                     n = 'output_{}_{}'.format(a, b)
#                     if SIG:
#                         self.fc_ws(ws=self.ws[2][b], name=n)
#                     else:
#                         self.fc_w2(ws=self.ws[2][b], name=n)
#                 else:
#                     n = 'output_{}'.format(a)
#                     if SIG:
#                         self.fc_ws(ws=self.ws[2][b], name=n, sig=False)
#                     else:
#                         self.fc_w2(ws=self.ws[2][b], name=n, relu=False)
#
#
def run_test(input_dic, sess, xy, te, cfg, mul=1):

    rst_dic = {}
    truth_dic = {}
    for a in range(mul):
        tr_pre_data = te.prepare(multi=-1, rd = False)
        run_data_1(tr_pre_data, input_dic, sess, xy, cfg, rst_dic, truth_dic)

    print(len(rst_dic))
    file_set = {}
    for id in rst_dic:
        if id[2] not in file_set:
            file_set[id[2]] = 0
        mx = max(id[:2])
        if mx>file_set[id[2]]:
            file_set[id[2]] = mx
        rt = np.array(rst_dic[id])
        rst_dic[id] = np.median(rt, axis=0)[-1]

    mtx = SortedDict()
    for f in file_set:
        for id in rst_dic:
            if id[2] == f and id[1]-id[0]==1:
                tr = truth_dic[id]
                mtx[id[0]] = Utils.create_M(tr/te.t_scale[:3])

    print(len(mtx))



#
# def avg_file_name(p):
#     basename = os.path.basename(p)
#     pathname = os.path.dirname(p)
#     return pathname + '_' + basename+'_avg.p'

if __name__ == '__main__':

    config_file = "config.json"

    cfg = Config(config_file)

    avg_file = avg_file_name(cfg.netFile)

    dt = 'te'

    if len(sys.argv)>1:
        dt = sys.argv[1]

    if dt=='te':
        te = DataSet(cfg.te_data, cfg)
    else:
        te = DataSet(cfg.tr_data, cfg)

    cfg.att = te.att
    te.avg_correction(avg_file)

    print("input attribute", cfg.att, "LR", cfg.lr,
          'feature', cfg.feature_len, 'add', cfg.add_len)

    inputs = {}

    Nout = cfg.num_output - cfg.num_output1
    setattr(cfg, 'Nout', Nout)

    # output = tf.placeholder(tf.float32, [None, cfg.num_output])
    output = tf.placeholder(tf.float32, [None, Nout])

    cfg.ref_node = cfg.nodes[0][-1]
    cfg.refs = np.ones(cfg.ref_node) #(np.array(range(cfg.ref_node)) + 1.0)/cfg.ref_node - 0.5
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
    for a in range(cfg.feature_len):
        n = 'output_{}'.format(a)
        if n in net.layers:
            xy[a] = net.layers['output_{}'.format(a)]
    print ('output', len(xy))

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
        rst_te = run_test(input_dic, sess, xy, te, cfg, mul)

        te.set_A_T(rst_te)

        te.save_data_2('te')
