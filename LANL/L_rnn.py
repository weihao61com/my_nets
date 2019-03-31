import sys
#from fc_dataset import *
import tensorflow as tf
import datetime
from sortedcontainers import SortedDict
from LANL_Utils import l_utils
import glob
import numpy as np

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))
from network import Network
from utils import Utils
#from fc_dataset import DataSet

def run_data(data, inputs, sess, xy, fname, cfg):
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


def run_data(data, c, inputs, sess, xy, filename, cfg):
    truth, features = l_utils.prepare_data(data, c)
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


if __name__ == '__main__':

    config_file = "rnn_config.json"

    test = None
    if len(sys.argv)>1:
        test = sys.argv[1]

    cfg = Utils.load_json_file(config_file)

    if test is None:
        locs = sorted(glob.glob(cfg['out_location'].format(HOME, '*')))
        data, att = l_utils.get_dataset(locs)
        print 'data att', att
        CV = cfg['CV']
    else:
        if test == 'te':
            te = DataSet([cfg.te_data[0]], cfg)
        else:
            te = DataSet([cfg.tr_data[0]], cfg)
        cfg.att = te.sz[1]

    lr0 = 1e-4
    iterations = 5
    loop = 1
    batch_size = 100
    netFile = cfg['netFile']

    cntn = False

    for c in range(CV):
        lr = lr0
        output = tf.placeholder(tf.float32, [None, 1])
        learning_rate = tf.placeholder(tf.float32, shape=[])

        dim = cfg['dim']
        n_att = cfg['n_att']
        layers = dim/n_att
        print 'CV', c, dim, n_att, layers

        cfg['ref_node'] = map(int, cfg['nodes0'].split(','))[-1]
        cfg['refs'] = np.ones(cfg['ref_node'])  # (np.array(range(cfg.ref_node)) + 1.0)/cfg.ref_node - 0.5
        cfg['refs'] = cfg['refs'].reshape((1, cfg['ref_node']))

        inputs = {}
        inputs[0] = tf.placeholder(tf.float32, [None, cfg['ref_node']])

        cfg['feature_len'] = layers
        for a in range(layers):
            inputs[a+1] = tf.placeholder(tf.float32, [None, n_att])

        input_dic = {}
        for a in range(layers + 1):
            input_dic['input_{}'.format(a)] = inputs[a]

        net = rNet(input_dic)
        net.real_setup(cfg, SIG=(cfg['SIG'] == 1))

        xy = net.layers['output']
        loss = tf.reduce_sum(tf.abs(tf.subtract(xy, output)))
        # loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))

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
                saver.restore(sess, cfg.netFile)
                run_test(input_dic, sess, xy, te, cfg)

            if cntn:
                saver.restore(sess, netFile.format(HOME, c))

            st1 = ''
            for a in range(iterations):

                te_loss = run_data(data[0], c+1, input_dic, sess, xy, '{}/tmp/te.csv'.format(HOME), cfg)
                tr_loss = run_data(data[0], -c-1, input_dic, sess, xy, '{}/tmp/tr.csv'.format(HOME), cfg)

                t1 = (datetime.datetime.now()-t00).seconds/3600.0
                str = "it: {0} {1:.3f} {2} {3} {4}".format(
                    a*loop/1000.0, t1, lr, tr_loss, te_loss)
                print str, st1

                t_loss = 0
                t_count = 0
                for dd in data:
                    truth, features = l_utils.prepare_data(dd, -c - 1, rd=True)
                    b0 = truth.reshape((len(truth), 1))
                    for lp in range(loop):
                        for d in range(0, len(truth), batch_size):
                            feed = {}
                            att = cfg['n_att']
                            length = features.shape[1] / att
                            for a in range(length):
                                feed[input_dic['input_{}'.format(a + 1)]] = features[d:d + batch_size, att * a:att * (a + 1)]
                            b_sz = len(features[d:d + batch_size, 0])
                            feed[input_dic['input_0']] = np.repeat(cfg['refs'], b_sz, axis=0)
                            feed[output] = b0[d:d + batch_size, :]
                            feed[learning_rate] = lr

                            _, A = sess.run([opt, loss], feed_dict=feed)
                            t_loss += A
                            t_count += len(truth[d:d + batch_size])
                st1 = '{}'.format(t_loss / t_count)

                saver.save(sess, netFile.format(HOME, c))

        tf.reset_default_graph()


