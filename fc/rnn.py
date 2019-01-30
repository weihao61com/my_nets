import sys
from fc_dataset import *
import tensorflow as tf
import datetime

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils
from fc_dataset import DataSet

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

class rNet_SIG(Network):

    def create_ws(self, n, ins, outs):
        print n, ins, outs
        w = self.make_var('weights_{}'.format(n), shape=[ins, outs])
        b = self.make_var('biases_{}'.format(n), shape=[outs])
        return [w,b]

    def parameters(self, cfg):

        self.ws = []

        # feature
        ws = []
        ins = cfg.ref_node + cfg.att
        nodes_in = cfg.nodes[0]
        for a in range(len(nodes_in)):
            ws.append(self.create_ws('feature_{}'.format(a), ins, nodes_in[a]))
            ins = nodes_in[a]
        self.ws.append(ws)

        # out
        ws = []
        nodes = cfg.nodes[1]
        for a in range(len(nodes)):
            ws.append(self.create_ws('out_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        ws.append(self.create_ws('out', ins, cfg.num_output))
        self.ws.append(ws)

    def setup(self):
        pass

    def real_setup(self, cfg, verbose=True):
        self.parameters(cfg)

        n = None
        a =0
        self.feed('input_0', 'input_1').\
            concat(1, name='inputs_{}'.format(a))
        for b in range(len(self.ws[0])):
            n = 'input_{}_{}'.format(a, b)
            self.fc_ws(ws=self.ws[0][b], name=n)

        ref_out = n
        for a in range(1, cfg.feature_len):
            inputs = 'input_{}'.format(a+1)
            n = 'input_{}_0'.format(a)
            self.feed(ref_out, inputs) \
                .concat(1, name=n)
            for b in range(len(self.ws[0])):
                n = 'input_{}_{}'.format(a, b)
                self.fc_ws(ws=self.ws[0][b], name=n)
                ref_out = n

            # final net a
            # a = 0
            if a < cfg.feature_len/2:
                continue

            self.feed(ref_out)
            for b in range(len(self.ws[1])):
                if b < len(self.ws[1])-1:
                    n = 'output_{}_{}'.format(a, b)
                    self.fc_ws(ws=self.ws[1][b], name=n)
                else:
                    n = 'output_{}'.format(a)
                    self.fc_ws(ws=self.ws[1][b], name=n, sig=False)

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
        ins = cfg.ref_node + cfg.att
        nodes_in = cfg.nodes[0]
        for a in range(len(nodes_in)):
            ws.append(self.create_ws('feature_{}'.format(a), ins, nodes_in[a]))
            ins = nodes_in[a]
        self.ws.append(ws)

        # out
        ws = []
        nodes = cfg.nodes[1]
        for a in range(len(nodes)):
            ws.append(self.create_ws('out_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        ws.append(self.create_ws('out', ins, cfg.num_output))
        self.ws.append(ws)

    def setup(self):
        pass

    def real_setup(self, cfg, verbose=True):
        self.parameters(cfg)

        n = None
        a =0
        self.feed('input_0', 'input_1').\
            concat(1, name='inputs_{}'.format(a))
        for b in range(len(self.ws[0])):
            n = 'input_{}_{}'.format(a, b)
            self.fc_w2(ws=self.ws[0][b], name=n)

        ref_out = n
        for a in range(1, cfg.feature_len):
            inputs = 'input_{}'.format(a+1)
            n = 'input_{}_0'.format(a)
            self.feed(ref_out, inputs) \
                .concat(1, name=n)
            for b in range(len(self.ws[0])):
                n = 'input_{}_{}'.format(a, b)
                self.fc_w2(ws=self.ws[0][b], name=n)
                ref_out = n

            # final net a
            # a = 0
            if a < cfg.feature_len - 2:
                continue

            self.feed(ref_out)
            for b in range(len(self.ws[1])):
                if b < len(self.ws[1])-1:
                    n = 'output_{}_{}'.format(a, b)
                    self.fc_w2(ws=self.ws[1][b], name=n)
                else:
                    n = 'output_{}'.format(a)
                    self.fc_w2(ws=self.ws[1][b], name=n, relu=False)

def run_test(input_dic, sess, xy, te, cfg):

    tr_pre_data = te.prepare(multi=-1)
    tr_loss, tr_median = run_data(tr_pre_data, input_dic, sess, xy, 'test', cfg)

    for a in range(len(tr_loss)):
        print a, tr_loss[a], tr_median[a]

    exit(0)


if __name__ == '__main__':

    config_file = "rnn_config.json"

    #if len(sys.argv)>1:
    #    config_file = sys.argv[1]

    test = None
    if len(sys.argv)>1:
        test = sys.argv[1]

    cfg = Config(config_file)

    if test is None:
        tr = DataSet(cfg.tr_data, cfg)
        te = DataSet(cfg.te_data, cfg, sub_sample=0.15)
        tr0 = DataSet([cfg.tr_data[0]], cfg, sub_sample=0.15)
        cfg.att = te.sz[1]
    else:
        if test == 'te':
            te = DataSet([cfg.te_data[0]], cfg)
        else:
            te = DataSet([cfg.tr_data[0]], cfg)
        cfg.att = te.sz[1]

    iterations = 10000
    loop = cfg.loop
    print "input attribute", cfg.att, "LR", cfg.lr, \
        'feature', cfg.feature_len, 'add', cfg.add_len

    inputs = {}

    output = tf.placeholder(tf.float32, [None, cfg.num_output])
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
    net.real_setup(cfg, verbose=False)

    xy = {}
    for a in range(cfg.feature_len):
        n = 'output_{}'.format(a)
        if n in net.layers:
            xy[a] = net.layers['output_{}'.format(a)]

    #ls = [] #[tf.reduce_sum(tf.square(tf.subtract(xy[0], output)))]
    loss = None

    for a in xy:
        if a >= cfg.feature_len/2:
            if loss is None:
                loss = tf.reduce_sum(tf.square(tf.subtract(xy[a], output)))
            else:
                loss = loss + tf.reduce_sum(tf.square(tf.subtract(xy[a], output)))
    #for x in range(1):
    #    ll = tf.reduce_sum(tf.square(tf.subtract(xy[x], output)))
    #    if loss is None:
    #        loss = ll
    #    else:
    #        loss = loss + ll
    #    ls.append(ll)

    opt = tf.train.AdamOptimizer(learning_rate=cfg.lr, beta1=0.9,
                    beta2=0.999, epsilon=0.00000001,
                    use_locking=False, name='Adam').\
        minimize(loss)
    # opt = tf.train.GradientDescentOptimizer(learning_rate=cfg.lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    t00 = datetime.datetime.now()

    with tf.Session() as sess:
        sess.run(init)

        if test is not None:
            saver.restore(sess, cfg.netTest)
            run_test(input_dic, sess, xy, te, cfg)

        if cfg.renetFile:
            saver.restore(sess, cfg.renetFile)

        str1 = ''
        for a in range(iterations):

            tr_pre_data = tr0.prepare()
            tr_loss, tr_median = run_data(tr_pre_data, input_dic, sess, xy, 'tr', cfg)

            te_pre_data = te.prepare()
            te_loss, te_median = run_data(te_pre_data, input_dic, sess, xy, 'te', cfg)

            t1 = datetime.datetime.now()
            str = "it: {0:.3f} {1:.3f}".format(a*loop/1000.0, (t1 - t00).total_seconds()/3600.0)
            s = 0
            while True:
                str += " {0:.3f} {1:.3f} {2:.3f} {3:.3f} ".format(tr_loss[s], te_loss[s], tr_median[s], te_median[s])
                s += 1
                if s==len(tr_loss):
                    break

            print str, str1

            tl3 = 0
            tl4 = 0
            tl5 = 0
            nt = 0
            att = cfg.att
            for _ in range(loop):
                tr_pre_data = tr.prepare(multi=1)

                while tr_pre_data:
                    for b in tr_pre_data:
                        total_length = len(b[0])
                        length = b[0].shape[1]/cfg.att
                        for c in range(0, total_length, cfg.batch_size):
                            feed = {}
                            n0 = 0
                            for a in range(length):
                                x = b[0][c:c + cfg.batch_size, cfg.att * a:cfg.att * (a + 1)]
                                feed[inputs[a + 1]] = x
                                n0 = x.shape[0]
                            feed[inputs[0]] = np.repeat(cfg.refs, n0, axis=0)
                            feed[output] = b[1][c:c + cfg.batch_size]

                            ll3,_= sess.run([loss, opt],feed_dict=feed)
                            tl3 += ll3
                            nt += n0
                    tr_pre_data = tr.get_next()
            str1 = "{0:.3f} ".format(tl3/nt)
            Utils.save_tf_data(saver, sess, cfg.netFile)


