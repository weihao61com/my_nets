import sys
import os
#from fc_dataset import *
import tensorflow as tf
import datetime
from sortedcontainers import SortedDict
from LANL_Utils import l_utils
import glob
import numpy as np
import pickle
import random
from scipy import fftpack, fft


HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))
from network import Network
from utils import Utils
#from fc_dataset import DataSet


def run_data_not(data, inputs, sess, xy, fname, cfg):
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
    for id in rst_dic:
        dst = np.array(rst_dic[id])
        result = np.median(dst, axis=0)
        results.append(result)
        truth.append(truth_dic[id])
        t = truth_dic[id]
        if random.random() < 0.2:
            r = np.linalg.norm(t - result)
            mm = result[-1]
            if len(mm)==3:
                fp.write('{},{},{},{},{},{},{}\n'.
                     format(t[0], mm[0], t[1], mm[1], t[2], mm[2], r))
            else:
                fp.write('{},{},{}\n'.
                         format(t[0], mm[0], r))
    fp.close()

    return Utils.calculate_stack_loss_avg(np.array(results), np.array(truth))


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
        ins = cfg['n_att']
        nodes = map(int, cfg['nodes2'].split(','))
        for a in range(len(nodes)):
            ws.append(self.create_ws('feature_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        self.ws.append(ws)

        # base
        ws = []
        ins += cfg['ref_node']
        nodes_in = map(int, cfg['nodes0'].split(','))
        for a in range(len(nodes_in)):
            ws.append(self.create_ws('base_{}'.format(a), ins, nodes_in[a]))
            ins = nodes_in[a]
        self.ws.append(ws)

        # out
        ws = []
        nodes = map(int, cfg['nodes1'].split(','))
        for a in range(len(nodes)):
            ws.append(self.create_ws('out_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        ws.append(self.create_ws('out', ins, 1))
        self.ws.append(ws)

    def setup(self):
        pass

    def real_setup(self, cfg, SIG=False):
        self.parameters(cfg)

        ref_out = 'input_0'
        for a in range(0, cfg['feature_len']):
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

            if a < cfg['feature_len']-1:
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
                    n = 'output'
                    if SIG:
                        self.fc_ws(ws=self.ws[2][b], name=n, sig=False)
                    else:
                        self.fc_w2(ws=self.ws[2][b], name=n, relu=False)


def create_T_F(y, x, t0):
    x = np.array(x)
    y = np.array(y)
    win = 300
    sz = len(x) / win
    b = x.reshape((win, sz)).T
    dd = []
    for a in range(win):
        d = abs(fft(b[:, a]))
        dd.append(d)

    dd = np.array(dd)
    d0 = np.mean(dd, 0)
    # d0 = np.std(dd, 0)
    # d0 = np.median(dd, 0)
    t = np.average(y) / t0
    # if t > 0.1 and t < 7.0:
    f = d0[1:sz / 2 + 1]
    return t, f


def generate_test_data(files, cfg):
    SEG = cfg['SEG']
    Ts = {}
    for f in files:
        x0, y0 = l_utils.load_data(f)
        t0 = y0[0]
        T = []
        print f, len(y0), t0
        for r in range(0, len(y0), l_utils.SEGMENT):
            x = x0[r:r+l_utils.SEGMENT]
            y = y0[r:r+l_utils.SEGMENT]
            t, _ = create_T_F(y, x, t0)
            F = []
            if len(y)==l_utils.SEGMENT:
                for rr in range(0, l_utils.SEGMENT, SEG/15):
                    if len(y[rr:rr+SEG]) == SEG:
                        # _, fs = create_T_F(y[rr:rr + SEG], x[rr:rr + SEG], t0)
                        _, fs = create_T_F(y[rr:rr + SEG], x[rr:rr + SEG], 1)
                        F.append(fs)
            if len(F)>0:
                T.append((t, F))
        Ts[f] = T
    return Ts


def get_id(filename):
    ids = os.path.basename(filename).split('.')[0]
    ids = ids.split('_')
    return ids[0]+'_'+ids[1]


def generate_data(file1, prob):

    ids = []
    for f in file1:
        ids.append(get_id(f))

    files = glob.glob(os.path.join(HOME, 'tmp0/L*.p'))
    T = []
    F = []
    random.shuffle(files)
    for f in files:
        id = get_id(f)
        if id in ids:
            if random.random() < prob:
                # print f
                with open(f, 'r') as fp:
                    A = pickle.load(fp)
                    T = T + A[0]
                    F = F + A[1]
            if prob > 1.0:
                ids.remove(id)

    T = np.array(T)
    F = np.array(F)
    # print 'GD', len(T)
    #    , np.min(T), np.max(T), np.average(T), np.std(T),\
    #    np.average(F), np.std(F), np.min(F), np.max(F)
    return T, F


def run_data(data, inputs, sess, xy, filename, cfg):
    truth = data[0]
    features = data[1]
    feed = {}
    b_sz = len(truth)
    att = cfg['n_att']
    length = features.shape[1]/att
    feed[inputs['input_0']] = np.repeat(cfg['refs'], b_sz, axis=0)
    for a in range(length):
        feed[inputs['input_{}'.format(a + 1)]] = features[:, att * a:att * (a + 1)]

    results = sess.run(xy, feed_dict=feed)[:, 0]

    if filename is not None:
        with open(filename, 'w') as fp:
            skip = int(len(truth)/2000)
            if skip==0:
                skip=1
            for a in range(len(truth)):
                if a%skip==0:
                    fp.write('{},{}\n'.format(truth[a], results[a]))

    return np.mean(np.abs(results-truth))


def run_test(file1, inputs, sess, xy, cfg, c=0):
    dd = generate_test_data(file1, cfg)
    fp = open('r.csv', 'w')

    e0 = []
    e1 = []
    for f in dd:
        data = dd[f]
        for data1 in data:

            feed = {}
            b_sz = len(data1[1])
            att = cfg['n_att']
            features = np.array(data1[1])
            #print features.shape
            length = features.shape[1]/att
            feed[inputs['input_0']] = np.repeat(cfg['refs'], b_sz, axis=0)
            for a in range(length):
                feed[inputs['input_{}'.format(a + 1)]] = features[:, att * a:att * (a + 1)]
            results = sess.run(xy, feed_dict=feed)[:, 0]

            st = np.std(results)
            mn = np.mean(results)
            md = np.median(results)
            e0.append(np.abs(mn-data1[0]))
            e1.append(np.abs(md-data1[0]))
            fp.write('{},{},{},{},{},{},{}\n'.format(c, f, data1[0], len(results), mn, md, st))
    fp.close()
    print np.mean(e0), np.mean(e1)

def gen_data(cfg, tmp='tmp'):
    files = glob.glob(os.path.join(cfg['location'].format(HOME), 'L_*.csv'))
    ids = l_utils.rdm_ids(files)

    file2 = []
    for f in ids:
        if ids[f] > -1:
            file2.append(f)

    gen = 5
    step = 4000
    SEG = cfg['SEG']
    for f in files:
        basename = os.path.basename(f)[:-4]
        basename = os.path.join(HOME, tmp, basename)
        x, y = l_utils.load_data(f)
        t0 = y[0]
        print f, len(x)/150000, basename, t0
        N = len(x) / step

        for g in range(gen):
            T = []
            F = []
            rps = np.random.randint(0, len(x) - l_utils.SEGMENT - 1, N)
            for r in rps:
                t, f = create_T_F(y[r:r + l_utils.SEGMENT], x[r:r + l_utils.SEGMENT], t0)
                T.append(t)
                F.append(f)
            fn = basename + '_{}.p'.format(g)
            print g, len(T)
            if len(T)>0:
                with open(fn, 'w') as fp:
                    pickle.dump((T, F), fp)


def sub_avg(data, mn, st):
    dd = data[1]
    for a in range(len(dd)):
        dd[a, :] = (dd[a, :]-mn)/st
    return data

def train(c, cfg, test=None):

    files = glob.glob(os.path.join(cfg['location'].format(HOME), 'L_*.csv'))
    ids = l_utils.rdm_ids(files)

    file1 = []
    file2 = []
    for f in ids:
        if ids[f] == c:
            file1.append(f)
        else:
            file2.append(f)

    if test is None:
        data2 = generate_data(file2, .1)
        data1 = generate_data(file1, 1.1)

    SEG = data1[1].shape[1]
    att = 25
    feature_len = SEG/att
    cfg['n_att'] = att

    lr = 1e-7
    cntn = 'cntn' in cfg
    iterations = 10000
    loop = 5
    batch_size = 100
    output = tf.placeholder(tf.float32, [None, 1])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    print 'CV, seg, att, feature_len', c, SEG, att, feature_len

    len_ref = map(int, cfg['nodes0'].split(','))[-1]
    cfg['ref_node'] = len_ref
    cfg['refs'] = np.ones(len_ref)
    cfg['refs'] = cfg['refs'].reshape((1, len_ref))
    netFile = cfg['netFile'].format(HOME, c)
    avg_file = os.path.dirname(netFile)+'_avg.p'
    if not cntn:
        mn = np.mean(data2[1], 0)
        st = np.std(data2[1], 0)
        with open(avg_file, 'w') as fp:
            pickle.dump((mn, st), fp)
    else:
        with open(avg_file, 'r') as fp:
            A = pickle.load(fp)
            mn = A[0]
            st = A[1]

    data1 = sub_avg(data1, mn, st)
    data2 = sub_avg(data2, mn, st)

    inputs = {0: tf.placeholder(tf.float32, [None, len_ref])}

    cfg['feature_len'] = feature_len
    for a in range(feature_len):
        inputs[a + 1] = tf.placeholder(tf.float32, [None, att])

    input_dic = {}
    for a in range(feature_len + 1):
        input_dic['input_{}'.format(a)] = inputs[a]


    net = rNet(input_dic)
    net.real_setup(cfg, SIG=(cfg['SIG'] == 1))

    xy = net.layers['output']
    loss = tf.reduce_sum(tf.abs(tf.subtract(xy, output)))
    # loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9,
                                 beta2=0.999, epsilon=0.00000001,
                                 use_locking=False, name='Adam'). \
        minimize(loss)
    # opt = tf.train.GradientDescentOptimizer(learning_rate=cfg.lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    t00 = datetime.datetime.now()

    with tf.Session() as sess:
        sess.run(init)

        if test is not None:
            saver.restore(sess, netFile)
            run_test(file1, input_dic, sess, xy,  cfg)
            exit(0)

        if cntn:
            saver.restore(sess, netFile)

        st1 = ''
        for a in range(iterations):

            te_loss = run_data(data1, input_dic, sess, xy, '{}/tmp/rte.csv'.format(HOME), cfg)
            tr_loss = run_data(data2, input_dic, sess, xy, '{}/tmp/rtr.csv'.format(HOME), cfg)

            t1 = (datetime.datetime.now() - t00).seconds / 3600.0
            str = "it: {0} {1:.3f} {2} {3} {4}".format(
                a * loop / 1000.0, t1, lr, tr_loss, te_loss)
            print str, st1

            t_loss = 0
            t_count = 0
            dd = generate_data(file2, 0.1)
            dd = sub_avg(dd, mn, st)

            truth = dd[0]
            features = dd[1]
            b0 = truth.reshape((len(truth), 1))
            for lp in range(loop):
                for d in range(0, len(truth), batch_size):
                    feed = {}
                    att = cfg['n_att']
                    length = features.shape[1] / att
                    for a in range(length):
                        feed[input_dic['input_{}'.format(a + 1)]] = features[d:d + batch_size,
                                                                    att * a:att * (a + 1)]
                    b_sz = len(features[d:d + batch_size, 0])
                    feed[input_dic['input_0']] = np.repeat(cfg['refs'], b_sz, axis=0)
                    feed[output] = b0[d:d + batch_size, :]
                    feed[learning_rate] = lr

                    _, A = sess.run([opt, loss], feed_dict=feed)
                    t_loss += A
                    t_count += len(truth[d:d + batch_size])
            st1 = '{} {}'.format(t_loss , t_count)

            saver.save(sess, netFile)

    tf.reset_default_graph()


if __name__ == '__main__':

    config_file = "rnn_config.json"

    test = None
    if len(sys.argv)>1:
        test = sys.argv[1]

    cfg = Utils.load_json_file(config_file)

    if test is None:
        gen_data(cfg)
        train(0, cfg)
    elif test =='t':
        train(0, cfg)
    elif test == '0':
        train(0, cfg, 't')
    elif test == 'g':
        gen_data(cfg, 'tmp')
