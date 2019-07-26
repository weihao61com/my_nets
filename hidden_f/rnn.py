import sys
# from fc_dataset import *
import tensorflow as tf
import datetime
from sortedcontainers import SortedDict
import numpy as np
import os
from dataset_h import DataSet, reverse
import pickle
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
            if not b[3][a] in rst_dic:
                rst_dic[b[3][a]] = []
            rst_dic[b[3][a]].append(result[:, a, :])
            truth_dic[b[3][a]] = b[1][a]

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
            if not b[3][a] in rst_dic:
                rst_dic[b[3][a]] = []
            rst_dic[b[3][a]].append(result[:, a, :])
            truth_dic[b[3][a]] = b[1][a], b[3][a]


def run_data(rst_dic, truth_dic, fname, cfg, Is):

    results = []
    truth = []

    filename = '/home/weihao/tmp/{}.csv'.format(fname)
    if sys.platform == 'darwin':
        filename = '/Users/weihao/tmp/{}.csv'.format(fname)
    fp = open(filename, 'w')
    rs = []
    list_dic = {}
    RR = []
    I = []
    t_scale = np.fromiter(map(float, cfg.t_scale.split(",")), dtype=np.float)
    for id in rst_dic:
        dst = np.array(rst_dic[id])
        result = np.median(dst, axis=0)
        # result = np.mean(dst, axis=0)
        #print(result, truth_dic[id])
        results.append(result)
        truth.append(truth_dic[id][0])
        t = truth_dic[id][0]
        imgs = truth_dic[id][1]
        result = result[-1]
        print(id, imgs)
        if imgs[0]<20:
            RR.append(Utils.create_M(result/t_scale[:3]))
            I.append(imgs)

        if len(t)==6 and imgs[0]>imgs[1]:
            t = reverse(t/t_scale)*t_scale
            result = reverse(result/t_scale)*t_scale
        dr = t - result
        r = np.linalg.norm(dr)
        rs.append(r*r)

        if random.random() < 1.2:
            if len(t) == 6 or imgs[0] < imgs[1]:

                mm = result
                for a in range(len(t)):
                    if a not in list_dic:
                        list_dic[a] = []
                    list_dic[a].append(t[a]-mm[a])
                    if a>0:
                        fp.write(',')
                    fp.write('{},{}'.format(t[a], mm[a]))
                fp.write(',{}\n'.format(r))

    P = {}
    P['RR'] = np.array(RR).transpose()
    P['Rgt'] = np.array(Is).transpose()
    P['I'] = np.array(I).transpose()
    filename = filename[:-4] + '.p'
    with open(filename, 'wb') as fp:
        pickle.dump(P, fp)

    for a in list_dic:
        vals = np.array(list_dic[a])
        md = np.median(abs(vals))
        avg = np.sqrt(np.mean(vals*vals))
        print('\t', a, md, avg)

    fp.close()
    rs = sorted(rs)
    length = len(rs)
    fp = open(filename+'.csv', 'w')
    for a in range(length):
        fp.write('{},{}\n'.format(float(a)/length, rs[a]))
    fp.close()

    return Utils.calculate_stack_loss_avg(np.array(results), np.array(truth), 0)


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


def run_test(input_dic, sess, xy, te, cfg, mul=1):

    rst_dic = {}
    truth_dic = {}
    for a in range(mul):
        tr_pre_data = te.prepare(multi=-1, rd = False)
        run_data_1(tr_pre_data, input_dic, sess, xy, cfg, rst_dic, truth_dic)


    tr_loss, tr_median = run_data(rst_dic, truth_dic, 'test', cfg, te.I)

    for a in range(len(tr_loss)):
        print(a, tr_loss[a], tr_median[a])

    exit(0)

def get_avg_file(tr, avg_file):
    av = None
    st = None
    nt = 0
    for id in tr.data:
        for ds in tr.data[id]:
            for d in ds[0]:
                if nt == 0:
                    av = d
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


def avg_file_name(p):
    basename = os.path.basename(p)
    pathname = os.path.dirname(p)
    return pathname + '_' + basename+'_avg.p'


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
        te = DataSet(cfg.te_data, cfg, sub_sample=0.15)
        tr0 = DataSet([cfg.tr_data[0]], cfg, sub_sample=0.1)
        tr.avg_correction(avg_file)
        tr0.avg_correction(avg_file)

    else:
        if test == 'te':
            te = DataSet([cfg.te_data[0]], cfg)
        else:
            te = DataSet([cfg.tr_data[0]], cfg)

    te.avg_correction(avg_file)
    iterations = 10000
    loop = cfg.loop
    print("input attribute", cfg.att, "LR", cfg.lr,
          'feature', cfg.feature_len, 'add', cfg.add_len)

    inputs = {}
    lr = cfg.lr
    learning_rate =  tf.compat.v1.placeholder(tf.float32, shape=[])

    Nout = cfg.num_output - cfg.num_output1
    setattr(cfg, 'Nout', Nout)

    # output = tf.placeholder(tf.float32, [None, cfg.num_output])
    output = tf.compat.v1.placeholder(tf.float32, [None, Nout])

    cfg.ref_node = list(cfg.nodes[0])[-1]
    cfg.refs = np.ones(cfg.ref_node) #(np.array(range(cfg.ref_node)) + 1.0)/cfg.ref_node - 0.5
    cfg.refs = cfg.refs.reshape((1, cfg.ref_node))
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

    loss = None
    last_loss = None
    As = []
    for a in xy:
        #if a<10:
        #    continue
        As.append(a)
        if cfg.L1==0:
            last_loss = tf.reduce_sum(tf.square(tf.subtract(xy[a], output)))
        else:
            last_loss = tf.reduce_sum(tf.abs(tf.subtract(xy[a], output)))
        if loss is None:
            loss = last_loss
        else:
            loss = loss + last_loss
    print(As)

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
            run_test(input_dic, sess, xy, te, cfg, mul)

        if cfg.renetFile:
            saver.restore(sess, cfg.renetFile)

        str1 = ''
        for a in range(iterations):

            t1 = datetime.datetime.now()
            str = "it: {0:.3f} {1:.3f} {2:4.2e}".\
                format(a*loop/1000.0, (t1 - t00).total_seconds()/3600.0, lr)

            tr_pre_data = tr0.prepare()
            tr_loss, tr_median = run_data_0(tr_pre_data, input_dic, sess, xy, 'tr', cfg)

            te_pre_data = te.prepare()
            te_loss, te_median = run_data_0(te_pre_data, input_dic, sess, xy, 'te', cfg)

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
            #tl4 = 0
            #tl5 = 0
            nt = 0
            att = cfg.att
            for _ in range(loop):
                tr_pre_data = tr.prepare(multi=cfg.multi)

                while tr_pre_data:
                    for b in tr_pre_data:
                        total_length = len(b[0])
                        length = int(b[0].shape[1]/cfg.att)
                        for c in range(0, total_length, cfg.batch_size):
                            feed = {learning_rate: lr}
                            n0 = 0
                            for a in range(length):
                                x = b[0][c:c + cfg.batch_size, cfg.att * a:cfg.att * (a + 1)]
                                feed[inputs[a + 1]] = x
                                n0 = x.shape[0]
                            feed[inputs[0]] = np.repeat(cfg.refs, n0, axis=0)
                            o = b[1][c:c + cfg.batch_size]
                            if len(o.shape) == 1:
                                o = o.reshape((len(o), 1))
                            feed[output] = o

                            ll3,_= sess.run([loss, opt],feed_dict=feed)
                            tl3 += ll3
                            nt += n0
                    tr_pre_data = tr.get_next(avg=avg_file)
                N_total += 1
                if N_total % cfg.INC_win == 0:
                    lr -= cfg.d_lr

            str1 = "{0:.3f} ".format(tl3/nt)
            Utils.save_tf_data(saver, sess, cfg.netFile)


