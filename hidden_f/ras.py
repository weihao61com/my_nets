import sys
# from fc_dataset import *
import tensorflow as tf
import datetime
from sortedcontainers import SortedDict
import numpy as np
import os
from dataset import DataSet
import cPickle
import random

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils, Config
from network import Network
#from fc_dataset import DataSet


def run_data_0(data, inputs, sess, xy, fname, cfg):
    att = cfg.att
    rst_dic = {}
    truth_dic = {}
    for b in data:
        length = b[0].shape[1]/att
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


def calculate_rst(rs):
    s = []
    for r in rs:
        s.append(r[0])
    s = np.array(s)
    return rs[0][1], np.mean(s, 0)

def run_data_1(data, inputs, sess, xy, cfg, rst_dic):

    sz = data[0].shape
    for a in xy:
        if a not in rst_dic:
            rst_dic[a] = {}

    for x in range(0, sz[0], cfg.batch_size):
        feed = {learning_rate: lr}
        b = data[0][x:x + cfg.batch_size]
        t = data[1][x:x + cfg.batch_size]
        i = data[2][x:x + cfg.batch_size]
        for y in range(sz[1]):
            feed[inputs['input_{}'.format(y + 1)]] = b[:, y, :]

        n0 = b.shape[0]
        feed[inputs['input_0']] = np.repeat(cfg.refs, n0, axis=0)

        r = sess.run(xy, feed_dict=feed)
        r = np.array(r.values())

        #for a in range(o.shape[0]):
        #    dd.append(t[a, cfg.out_offset:] - opt_out[:, a])

        for a in rst_dic:
            for b in range(len(i)):
                tr = t[b, a, :]
                id = i[b][a-cfg.out_offset][0]
                rs = r[a-cfg.out_offset, b, :]
                if id not in rst_dic[a]:
                    rst_dic[a][id] = []
                rst_dic[a][id].append((rs, tr))
        #
    # M = []
    # L = []
    # for a in xy:
    #     results = []
    #     truth = []
    #     rst = rst_dic[a]
    #     for id in rst:
    #         rt = rst_dic[a][id]
    #         t, r = calculate_rst(rt)
    #         results.append(r)
    #         truth.append(t)
    #     m, l = Utils.calculate_stack_loss_ras(np.array(results), np.array(truth), 0)
    #     M.append(m)
    #     L.append(l)
    #     print a, m, l
    #
    # return M, L


def run_data(rst_dic, fname):
    filename = '/home/weihao/tmp/{}.csv'.format(fname)
    if sys.platform == 'darwin':
        filename = '/Users/weihao/tmp/{}.csv'.format(fname)
    a_last = rst_dic.keys()[-1]

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
        print n, ins, outs
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
        if cfg.fc_Nout>0:
            Nout = cfg.fc_Nout*2
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


def run_test(input_dic, sess, xy, te, cfg, mul=1):

    rst_dic = {}
    for a in range(mul):
        tr_pre_data = te.prepare_ras(multi=-1, rd=False)
        run_data_1(tr_pre_data, input_dic, sess, xy, cfg, rst_dic)

    tr_loss, tr_median = run_data(rst_dic, 'test')

    for a in range(len(tr_loss)):
        print a, tr_loss[a], tr_median[a]

    exit(0)

def get_avg_file(tr, avg_file):
    av = None
    st = None
    nt = 0
    for id in tr.rasd.features:
        features = tr.rasd.features[id]
        for img_id in features:
            feature = np.array(features[img_id][0])
            if nt == 0:
                av = np.mean(np.array(feature), 0)
                st = np.std(np.array(feature), 0)
            else:
                av += np.mean(np.array(feature), 0)
                st += np.std(np.array(feature), 0)
            nt += 1
    av /= nt
    st /= nt
    print "Saving averages:", avg_file
    for a in range(len(av)):
        print a, av[a], st[a]
        if st[a]<1e-9:
            st[a] = 1

    with open(avg_file, 'w') as fp:
        cPickle.dump((av,st), fp)

    return


def avg_file_name(p, tag ='avg'):
    basename = os.path.basename(p)
    pathname = os.path.dirname(p)
    return pathname + '_' + basename+'_{}.p'.format(tag)


if __name__ == '__main__':

    config_file = "config.json"

    #if len(sys.argv)>1:
    #    config_file = sys.argv[1]

    test = None
    if len(sys.argv)>1:
        test = sys.argv[1]

    cfg = Config(config_file)

    avg_file = avg_file_name(cfg.netFile)
    if test is None:
        tr = DataSet(cfg.tr_data, cfg)
        get_avg_file(tr, avg_file)
        if cfg.fc_Nout>0:
            tr.init_truth(cfg.fc_Nout)
    else:
        if test == 'te':
            tr = DataSet([cfg.te_data[0]], cfg)
        else:
            tr = DataSet([cfg.tr_data[0]], cfg)

    tr.avg_correction2(avg_file)
    iterations = 10000
    loop = cfg.loop
    print "input attribute", cfg.att, "LR", cfg.lr, \
        'feature', cfg.feature_len, 'add', cfg.add_len

    inputs = {}
    output = {}
    lr = cfg.lr
    learning_rate = tf.placeholder(tf.float32, shape=[])

    cfg.Nout = cfg.fc_Nout*2
    cfg.Nout = 3

    cfg.ref_node = cfg.fc_nodes[0][-1]
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
    if cfg.fc_Nout>0:
        Nout = cfg.fc_Nout*2
    for a in range(cfg.feature_len):
        n = 'output_{}'.format(a)
        if n in net.layers:
            xy[a] = net.layers['output_{}'.format(a)]
            output[a] = tf.placeholder(tf.float32, [None, Nout])
    print 'output count:', len(xy)

    loss = None
    last_loss = None
    for a in xy:
        print a,
        if cfg.L1==0:
            last_loss = tf.reduce_sum(tf.square(tf.subtract(xy[a], output[a])))
        else:
            last_loss = tf.reduce_sum(tf.abs(tf.subtract(xy[a], output[a])))
        if loss is None:
            loss = last_loss
        else:
            loss = loss + last_loss
    print

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
            saver.restore(sess, cfg.renetFile)

        str1 = ''
        for a in range(iterations):

            t1 = datetime.datetime.now()
            str = "it: {0:.3f} {1:.3f} {2:4.2e}".\
                format(a*loop/1000.0, (t1 - t00).total_seconds()/3600.0, lr)

            print str, str1

            if lr<1e-9:
                break

            tl3 = 0
            to3 = 0
            r3 = 0
            nt = 0
            att = cfg.att
            for _ in range(loop):
                tr_pre_data = tr.prepare_ras(multi=cfg.multi)
                while tr_pre_data:
                    sz = tr_pre_data[0].shape
                    length = sz[1]

                    for x in range(0, sz[0], cfg.batch_size):
                        feed = {learning_rate: lr}
                        b = tr_pre_data[0][x:x+cfg.batch_size]
                        o = tr_pre_data[1][x:x+cfg.batch_size]
                        ids = tr_pre_data[2][x:x+cfg.batch_size]

                        n0 = b.shape[0]
                        for y in range(length):
                            feed[inputs[y + 1]] = b[:, y, :]
                            if y in output:
                                feed[output[y]] = o[:, y, :]
                        feed[inputs[0]] = np.repeat(cfg.refs, n0, axis=0)

                        # opt_out = sess.run(xy, feed_dict=feed)
                        ll3, opt_out, _ = sess.run([loss, xy, opt], feed_dict=feed)
                        opt_out = np.array(opt_out.values())
                        dd = []
                        o = o[:, cfg.out_offset:, :]
                        for a in range(o.shape[0]):
                            dd.append(o[a]-opt_out[:,a])
                        n1 = np.linalg.norm(o)
                        n2 = 1
                        lr0 = lr*1000
                        n1 = 10
                        if cfg.fc_Nout>0:
                            do = o - lr0 * np.array(dd)
                            n2 = np.linalg.norm(do)
                            do *= n1/n2
                            tr.updates(do, ids)
                        dd = np.array(dd)
                        dd = dd*dd
                        dd = dd[:, -1, :].sum()
                        to3 += n2*n0
                        tl3 += ll3
                        r3 += dd
                        nt += n0
                    tr_pre_data = tr.get_next(avg=avg_file)
            N_total += 1
            if N_total % cfg.INC_win == 0:
                lr -= cfg.d_lr

            tl3 /= nt
            to3 /= nt
            r3 /= nt
            str1 = "{0:.4f} {1:.4f}  {2:.5f} {3:.5f}".format(tl3, to3, r3, n1/n2)
            Utils.save_tf_data(saver, sess, cfg.netFile)


