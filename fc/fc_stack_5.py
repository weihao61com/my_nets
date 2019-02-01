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

def run_data_stack_avg3(data, inputs, sess, xy, fname, att, step=1):
    rst_dic = {}
    truth_dic = {}
    for b in data:
        length = b[0].shape[1]/att
        feed = {}
        for a in range(length):
            feed[inputs['input_{}'.format(a)]] = b[0][:, att * a:att * (a + 1)]
        result = []
        for a in range(0, length+1, step):
            r = sess.run(xy[a], feed_dict=feed)
            result.append(r)
        result = np.array(result)
        for a in range(len(b[2])):
            if not b[2][a] in rst_dic:
                rst_dic[b[2][a]] = []
            rst_dic[b[2][a]].append(result[:, a, :])
            truth_dic[b[2][a]] = b[1][a]

    results = []
    stds = []
    truth = []

    filename = '/home/weihao/tmp/{}.csv'.format(fname)
    if sys.platform == 'darwin':
        filename = '/Users/weihao/tmp/{}.csv'.format(fname)
    fp = open(filename, 'w')
    for id in rst_dic:
        dst = np.array(rst_dic[id])
        result = np.median(dst, axis=0)
        std = np.std(dst, axis=0)
        results.append(result)
        stds.append(std)
        truth.append(truth_dic[id])
        t = truth_dic[id]
        if random.random() < 0.2:
            mm = result[-1]
            r = np.linalg.norm(t - mm)
            if len(mm)==3:
                fp.write('{},{},{},{},{},{},{}\n'.
                     format(t[0], mm[0], t[1], mm[1], t[2], mm[2], r))
            else:
                fp.write('{},{},{}\n'.
                         format(t[0], mm[0], r))
    fp.close()
    s = np.array(stds)
    std = np.mean(np.linalg.norm(s, axis=2), axis=0)

    return Utils.calculate_stack_loss_avg(np.array(results), np.array(truth))

class StackNet5_SIG(Network):

    def create_ws(self, n, ins, outs):
        print n, ins, outs
        w = self.make_var('weights_{}'.format(n), shape=[ins, outs])
        b = self.make_var('biases_{}'.format(n), shape=[outs])
        return [w,b]

    def parameters(self, cfg):

        self.ws = []

        assert(cfg.nodes[1][-1] == cfg.nodes[2][-1])

        # feature
        ws = []
        ins = cfg.att
        nodes_in = cfg.nodes[0]
        for a in range(len(nodes_in)):
            ws.append(self.create_ws('feature_{}'.format(a), ins, nodes_in[a]))
            ins = nodes_in[a]
        self.ws.append(ws)

        # average
        ws = []
        ins = nodes_in[-1] * cfg.feature_len
        nodes = cfg.nodes[1]
        for a in range(len(nodes)):
            ws.append(self.create_ws('average_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        self.ws.append(ws)

        # stack
        ws = []
        ins = nodes[-1] + nodes_in[-1]
        nodes = cfg.nodes[2]
        for a in range(len(nodes)):
            ws.append(self.create_ws('stacker_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        self.ws.append(ws)

        # output
        ws = []
        ins = nodes[-1]
        nodes = cfg.nodes[3]
        nodes.append(cfg.num_output)
        for a in range(len(nodes)):
            ws.append(self.create_ws('outputs_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        self.ws.append(ws)

    def setup(self):
        pass

    def real_setup(self, cfg, verbose=True):
        self.parameters(cfg)

        # inputs nets
        all_inputs = []
        for a in range(cfg.feature_len + cfg.add_len):
            self.feed('input_{}'.format(a))
            n = None
            for b in range(len(self.ws[0])):
                n = 'input_{}_{}'.format(a, b)
                self.fc_ws(ws=self.ws[0][b], name=n)
            all_inputs.append(n)

        # average net
        self.feed(*all_inputs[:cfg.feature_len]).concat(1, name='avg_inputs')
        ref_out = None
        for b in range(len(self.ws[1])):
            ref_out = 'avg_input_{}'.format(b)
            self.fc_ws(ws=self.ws[1][b], name=ref_out)

        # final net 0
        self.feed(ref_out)
        a = 0
        for b in range(len(self.ws[3])):
            if b < len(self.ws[3])-1:
                n = 'output_{}_{}'.format(a, b)
                self.fc_ws(ws=self.ws[3][b], name=n)
            else:
                n = 'output_{}'.format(a)
                self.fc_ws(ws=self.ws[3][b], name=n, sig=False)

        # stack 1-n
        for a in range(cfg.feature_len + cfg.add_len):
            self.feed(ref_out, all_inputs[a]) \
                .concat(1, name='stack_inputs_{}'.format(a))
            for b in range(len(self.ws[2])):
                ref_out = 'stack_input_{}_{}'.format(a, b)
                self.fc_ws(ws=self.ws[2][b], name=ref_out)

            # final net a
            self.feed(ref_out)
            for b in range(len(self.ws[3])):
                if b < len(self.ws[3])-1:
                    n = 'output_{}_{}'.format(a, b)
                    self.fc_ws(ws=self.ws[3][b], name=n)
                else:
                    n = 'output_{}'.format(a+1)
                    self.fc_ws(ws=self.ws[3][b], name=n, sig=False)

class StackNet5(Network):

    def create_ws(self, n, ins, outs):
        print n, ins, outs
        w = self.make_var('weights_{}'.format(n), shape=[ins, outs])
        b = self.make_var('biases_{}'.format(n), shape=[outs])
        return [w,b]

    def parameters(self, cfg):

        self.ws = []

        assert(cfg.nodes[1][-1] == cfg.nodes[2][-1])

        # feature
        ws = []
        ins = cfg.att
        nodes_in = cfg.nodes[0]
        for a in range(len(nodes_in)):
            ws.append(self.create_ws('feature_{}'.format(a), ins, nodes_in[a]))
            ins = nodes_in[a]
        self.ws.append(ws)

        # average
        ws = []
        ins = nodes_in[-1] * cfg.feature_len
        nodes = cfg.nodes[1]
        for a in range(len(nodes)):
            ws.append(self.create_ws('average_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        self.ws.append(ws)

        # stack
        ws = []
        ins = nodes[-1] + nodes_in[-1]
        nodes = cfg.nodes[2]
        for a in range(len(nodes)):
            ws.append(self.create_ws('stacker_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        self.ws.append(ws)

        # output
        ws = []
        ins = nodes[-1]
        nodes = cfg.nodes[3]
        nodes.append(cfg.num_output)
        for a in range(len(nodes)):
            ws.append(self.create_ws('outputs_{}'.format(a), ins, nodes[a]))
            ins = nodes[a]
        self.ws.append(ws)

    def setup(self):
        pass

    def real_setup(self, cfg, verbose=True):
        self.parameters(cfg)

        # inputs nets
        all_inputs = []
        for a in range(cfg.feature_len + cfg.add_len):
            self.feed('input_{}'.format(a))
            n = None
            for b in range(len(self.ws[0])):
                n = 'input_{}_{}'.format(a, b)
                self.fc_w2(ws=self.ws[0][b], name=n)
            all_inputs.append(n)

        # average net
        self.feed(*all_inputs[:cfg.feature_len]).concat(1, name='avg_inputs')
        ref_out = None
        for b in range(len(self.ws[1])):
            ref_out = 'avg_input_{}'.format(b)
            self.fc_w2(ws=self.ws[1][b], name=ref_out)

        # final net 0
        self.feed(ref_out)
        a = 0
        for b in range(len(self.ws[3])):
            if b < len(self.ws[3])-1:
                n = 'output_{}_{}'.format(a, b)
                self.fc_w2(ws=self.ws[3][b], name=n)
            else:
                n = 'output_{}'.format(a)
                self.fc_w2(ws=self.ws[3][b], name=n, relu=False)

        # stack 1-n
        for a in range(cfg.feature_len + cfg.add_len):
            self.feed(ref_out, all_inputs[a]) \
                .concat(1, name='stack_inputs_{}'.format(a))
            for b in range(len(self.ws[2])):
                ref_out = 'stack_input_{}_{}'.format(a, b)
                self.fc_w2(ws=self.ws[2][b], name=ref_out)

            # final net a
            self.feed(ref_out)
            for b in range(len(self.ws[3])):
                if b < len(self.ws[3])-1:
                    n = 'output_{}_{}'.format(a, b)
                    self.fc_w2(ws=self.ws[3][b], name=n)
                else:
                    n = 'output_{}'.format(a+1)
                    self.fc_w2(ws=self.ws[3][b], name=n, relu=False)


def run_test(input_dic, sess, xy, te):

    att = te.sz[1]
    tr_pre_data = te.prepare(multi=-1)
    tr_loss, tr_median = run_data_stack_avg3(tr_pre_data, input_dic, sess, xy, 'test', att)

    for a in range(len(tr_loss)):
        print a, tr_loss[a], tr_median[a]

    exit(0)


if __name__ == '__main__':

    config_file = "config_stack_5.json"

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    test = None
    if len(sys.argv)>2:
        test = sys.argv[2]

    cfg = Config(config_file)

    if test is None:
        tr = DataSet(cfg.tr_data, cfg)
        te = DataSet(cfg.te_data, cfg, sub_sample=.5)
        tr0 = DataSet([cfg.tr_data[0]], cfg, sub_sample=.5)

        cfg.att = te.sz[1]
    else:
        if test == 'te':
            te = DataSet([cfg.te_data[0]], cfg)
        else:
            te = DataSet([cfg.tr_data[0]], cfg)
        cfg.att = te.sz[1]

    iterations = 10000
    loop = cfg.loop
    print "input attribute", cfg.att, "LR", cfg.lr, 'feature', cfg.feature_len

    inputs = {}

    output = tf.placeholder(tf.float32, [None, cfg.num_output])
    for a in range(cfg.feature_len + cfg.add_len):
        inputs[a] = tf.placeholder(tf.float32, [None, cfg.att])

    input_dic = {}
    for a in range(cfg.feature_len+cfg.add_len):
        input_dic['input_{}'.format(a)] = inputs[a]

    net = StackNet5(input_dic)
    net.real_setup(cfg, verbose=False)

    xy = {}
    for a in range(cfg.feature_len+cfg.add_len+ 1):
        xy[a] = net.layers['output_{}'.format(a)]

    ls = [] #[tf.reduce_sum(tf.square(tf.subtract(xy[0], output)))]
    loss = None
    for x in range(0, cfg.feature_len+cfg.add_len+1):
        ll = tf.reduce_sum(tf.square(tf.subtract(xy[x], output)))
        if loss is None:
            loss = ll
        else:
            loss = loss + ll
        ls.append(ll)

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
            saver.restore(sess, cfg.netFile)
            run_test(input_dic, sess, xy, te)

        if cfg.renetFile:
            saver.restore(sess, cfg.renetFile)

        str1 = ''
        for a in range(iterations):

            tr_pre_data = tr0.prepare()
            tr_loss, tr_median = run_data_stack_avg3(tr_pre_data, input_dic, sess, xy, 'tr', cfg.att, step=cfg.feature_len/2)

            te_pre_data = te.prepare()
            te_loss, te_median = run_data_stack_avg3(te_pre_data, input_dic, sess, xy, 'te', cfg.att, step=cfg.feature_len/2)

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
            for _ in range(loop):
                tr_pre_data = tr.prepare(multi=cfg.multi)

                while tr_pre_data:
                    for b in tr_pre_data:
                        total_length = len(b[0])
                        for c in range(0, total_length, cfg.batch_size):
                            feed = {}
                            for a in range(cfg.feature_len+cfg.add_len):
                                feed[input_dic['input_{}'.format(a)]] = \
                                    b[0][c:c + cfg.batch_size, cfg.att * a:cfg.att * (a + 1)]
                            feed[output] = b[1][c:c + cfg.batch_size]
                            idx = cfg.feature_len-1
                            ll3,ll4,ll5,_= sess.run([ls[0], ls[idx], ls[-1], opt],
                                                      feed_dict=feed)
                            tl3 += ll3
                            tl4 += ll4
                            tl5 += ll5
                            nt += len(b[0][c:c + cfg.batch_size])
                    tr_pre_data = tr.get_next()
            str1 = "{0:.3f} {1:.3f} {2:.3f}".format(tl3/nt, tl4/nt, tl5/nt)
            Utils.save_tf_data(saver, sess, cfg.netFile)


