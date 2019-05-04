import sys
from fc_dataset import *
import tensorflow as tf
import datetime
from sortedcontainers import SortedDict

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
    rs = []
    for id in rst_dic:
        dst = np.array(rst_dic[id])
        result = np.median(dst, axis=0)
        results.append(result)
        truth.append(truth_dic[id])
        t = truth_dic[id]
        if random.random() < 0.2:
            r = np.linalg.norm(t - result[-1])
            rs.append(r)
            mm = result[-1]
            if len(mm)==3:
                fp.write('{},{},{},{},{},{},{}\n'.
                     format(t[0], mm[0], t[1], mm[1], t[2], mm[2], r))
            else:
                fp.write('{},{},{}\n'.
                         format(t[0], mm[0], r))
    fp.close()
    rs = sorted(rs)
    length = len(rs)
    fp = open(filename+'.csv', 'w')
    for a in range(length):
        fp.write('{},{}\n'.format(float(a)/length, rs[a]))
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
        ws.append(self.create_ws('out', ins, cfg.num_output))
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

            #if a < cfg.feature_len/2:
            #    continue

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


def run_test(input_dic, sess, xy, te, cfg, mul=-1):

    tr_pre_data = te.prepare(multi=mul)
    tr_loss, tr_median = run_data(tr_pre_data, input_dic, sess, xy, 'test', cfg)

    for a in range(len(tr_loss)):
        print a, tr_loss[a], tr_median[a]

    exit(0)

def get_avg_file(tr, avg_file):
    av = None
    st = None
    nt = 0
    for d in tr.data[0]:
        if nt == 0:
            av = np.sum(d[0], 0)
            st = np.sum(d[0]*d[0], 0)
        else:
            av += np.sum(d[0], 0)
            st += np.sum(d[0] * d[0], 0)
        nt += d[0].shape[0]
    av /= nt
    st /= nt
    st = np.sqrt(st - av*av)
    print "Saving averages:", avg_file
    for a in range(len(av)):
        print a, av[a], st[a]

    with open(avg_file, 'w') as fp:
        pickle.dump((av,st), fp)

    return


def avg_file_name(p):
    basename = os.path.basename(p)
    pathname = os.path.dirname(p)
    return pathname + '_' + basename+'_avg.p'

if __name__ == '__main__':

    config_file = "rnn_config.json"

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
        cfg.att = te.sz[1]
        tr.avg_correction(avg_file)
        tr0.avg_correction(avg_file)

    else:
        if test == 'te':
            te = DataSet([cfg.te_data[0]], cfg)
        else:
            te = DataSet([cfg.tr_data[0]], cfg)
        cfg.att = te.sz[1]

    te.avg_correction(avg_file)
    iterations = 10000
    loop = cfg.loop
    print "input attribute", cfg.att, "LR", cfg.lr, \
        'feature', cfg.feature_len, 'add', cfg.add_len

    inputs = {}
    lr = cfg.lr
    learning_rate = tf.placeholder(tf.float32, shape=[])

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
    net.real_setup(cfg, SIG=(cfg.SIG==1))

    xy = SortedDict()
    for a in range(cfg.feature_len):
        n = 'output_{}'.format(a)
        if n in net.layers:
            xy[a] = net.layers['output_{}'.format(a)]
    print 'output', len(xy)

    loss = None
    last_loss = None
    for a in xy:
        #if a<10:
        #    continue
        last_loss = tf.reduce_sum(tf.square(tf.subtract(xy[a], output)))
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
            saver.restore(sess, cfg.netFile)
            run_test(input_dic, sess, xy, te, cfg)

        if cfg.renetFile:
            saver.restore(sess, cfg.renetFile)

        str1 = ''
        for a in range(iterations):

            t1 = datetime.datetime.now()
            str = "it: {0:.3f} {1:.3f} {2:4.2e}".\
                format(a*loop/1000.0, (t1 - t00).total_seconds()/3600.0, lr)

            tr_pre_data = tr0.prepare()
            tr_loss, tr_median = run_data(tr_pre_data, input_dic, sess, xy, 'tr', cfg)

            te_pre_data = te.prepare()
            te_loss, te_median = run_data(te_pre_data, input_dic, sess, xy, 'te', cfg)

            s = -1
            while True:
                s += len(tr_loss)/2
                str += " {0:.3f} {1:.3f} {2:.3f} {3:.3f} ".format(tr_loss[s], te_loss[s], tr_median[s], te_median[s])
                if s==len(tr_loss)-1:
                    break

            # tr_pre_data = tr0.prepare(multi=-1)
            # tr_loss, tr_median = run_data(tr_pre_data, input_dic, sess, xy, 'tr', cfg)
            #
            # te_pre_data = te.prepare(multi=-1)
            # te_loss, te_median = run_data(te_pre_data, input_dic, sess, xy, 'te', cfg)
            #
            # s = -1
            # while True:
            #     s += len(tr_loss)/2
            #     str += " {0:.3f} {1:.3f} {2:.3f} {3:.3f} ".format(tr_loss[s], te_loss[s], tr_median[s], te_median[s])
            #     if s==len(tr_loss)-1:
            #         break

            print str, str1

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
                        length = b[0].shape[1]/cfg.att
                        for c in range(0, total_length, cfg.batch_size):
                            feed = {learning_rate: lr}
                            n0 = 0
                            for a in range(length):
                                x = b[0][c:c + cfg.batch_size, cfg.att * a:cfg.att * (a + 1)]
                                feed[inputs[a + 1]] = x
                                n0 = x.shape[0]
                            feed[inputs[0]] = np.repeat(cfg.refs, n0, axis=0)
                            feed[output] = b[1][c:c + cfg.batch_size]

                            ll3,_= sess.run([last_loss, opt],feed_dict=feed)
                            tl3 += ll3
                            nt += n0
                    tr_pre_data = tr.get_next(avg=avg_file)
                N_total += 1
                if N_total % cfg.INC_win == 0:
                    lr -= cfg.d_lr
            if lr<1e-6:
                break

            str1 = "{0:.3f} ".format(tl3/nt)
            Utils.save_tf_data(saver, sess, cfg.netFile)


