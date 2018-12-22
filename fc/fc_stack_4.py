import sys
from fc_dataset import *
import tensorflow as tf
import datetime
import fc_const

from utils import Utils
from fc_dataset import DataSet


class StackNet4(Network):

    def setup(self):
        pass


    def parameters(self, dim_input=4, dim_output=3, dim_ref=128):

        self.dim_ref = dim_ref
        self.dim_output = dim_output

        self.dim0 = dim_input + dim_ref
        self.out0 = self.dim_inter[0]
        self.weights0 = self.make_var('weights0', shape=[self.dim0, self.out0])
        self.biases0 = self.make_var('biases0', [self.out0])

        self.dim1 = self.out0
        self.out1 = self.dim_inter[1]
        self.weights1 = self.make_var('weights1', shape=[self.dim1, self.out1])
        self.biases1 = self.make_var('biases1', [self.out1])
        #self.out1 = self.out0

        self.dim2 = self.out1
        self.out2 = dim_ref
        self.weights2 = self.make_var('weights2', shape=[self.dim2, self.out2])
        self.biases2 = self.make_var('biases2', [self.out2])

        self.dim3 = self.out2
        self.out3 = self.dim_output
        self.weights3 = self.make_var('weights3', shape=[self.dim3, self.out3])
        self.biases3 = self.make_var('biases3', [self.out3])

    def real_setup(self, stack, ns, num_out=3, att=4, verbose=True):
        self.stack = stack
        self.dim_inter = ns[2]
        nodes = ns[0]
        ref_dim = ns[1][0]

        self.parameters(dim_output=num_out, dim_ref=ref_dim, dim_input=att)

        self.feed('input0')
        for a in range(len(nodes)):
            name = 'fc_0{}'.format(a)
            self.fc(nodes[a], name=name)
        self.fc(self.dim_ref, name='fc1')
        self.fc(num_out, relu=False, name='output0')

        ref_out_name = 'fc1'
        for a in range(self.stack):
            input_name = 'input{}'.format(a + 1)
            ic_name = 'ic{}_in'.format(a)
            ifc0_name = 'ifc0{}_in'.format(a)
            ifc1_name = 'ifc1{}_in'.format(a)
            ifc2_name = 'ifc2{}_in'.format(a)
            output_name = 'output{}'.format(a + 1)

            (self.feed(input_name, ref_out_name)
             .concat(1, name=ic_name)
             .fc_w(name=ifc0_name,
                   weights=self.weights0,
                   biases=self.biases0)
             .fc_w(name=ifc1_name,
                   weights=self.weights1,
                   biases=self.biases1)
             .fc_w(name=ifc2_name,
                   weights=self.weights2,
                   biases=self.biases2)
             .fc_w(name=output_name, relu=False,
                   weights=self.weights3,
                   biases=self.biases3)
             )

            ref_out_name = ifc2_name

        for b in range(self.stack):
            a = b + self.stack
            input_name = 'input{}'.format(a + 1)
            ic_name = 'ic{}_in'.format(a)
            ifc0_name = 'ifc0{}_in'.format(a)
            ifc1_name = 'ifc1{}_in'.format(a)
            ifc2_name = 'ifc2{}_in'.format(a)
            output_name = 'output{}'.format(a + 1)

            (self.feed(input_name, ref_out_name)
             .concat(1, name=ic_name)
             .fc_w(name=ifc0_name,
                   weights=self.weights0,
                   biases=self.biases0)
             .fc_w(name=ifc1_name,
                   weights=self.weights1,
                   biases=self.biases1)
             .fc_w(name=ifc2_name,
                   weights=self.weights2,
                   biases=self.biases2)
             .fc_w(name=output_name, relu=False,
                   weights=self.weights3,
                   biases=self.biases3)
             )

            ref_out_name = ifc2_name
        print 'stack', self.dim_inter
        print 'base', nodes, self.dim_ref

        if verbose:
            print("number of layers = {}".format(len(self.layers)))
            for l in sorted(self.layers.keys()):
                print l, self.layers[l].get_shape()

def run_data_avg4(data, input_dic, sess, xy, fname):
    rst_dic = {}
    truth_dic = {}
    length = len(xy)
    for b in data:
        sz = b[0].shape[1]/2
        feed = {input_dic['input0']: b[0][:, :sz]}
        for d in range(cfg.feature_len * 2):
            feed[input_dic['input{}'.format(d + 1)]] = \
                b[0][:, att * d: att * (d + 1)]
        result = []
        for a in range(length):
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

    for id in rst_dic:
        dst = np.array(rst_dic[id])
        result = np.median(dst, axis=0)
        results.append(result)
        truth.append(truth_dic[id])
        t = truth_dic[id]

    return Utils.calculate_stack_loss_avg(np.array(results), np.array(truth))


if __name__ == '__main__':

    config_file = "config_stack_4.json"

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    cfg = Config(config_file)

    renetFile = cfg.renetFile

    tr = DataSet(cfg.tr_data, cfg.memory_size, cfg.feature_len*2)
    te = DataSet(cfg.te_data, cfg.memory_size, cfg.feature_len*2)
    tr.set_t_scale(cfg.t_scale)
    te.set_t_scale(cfg.t_scale)
    tr.set_num_output(cfg.num_output)
    te.set_num_output(cfg.num_output)
    step = cfg.batch_size

    att = te.sz[1]
    iterations = int(1e6)
    print "input attribute", att, "LR", cfg.lr, 'feature', cfg.feature_len

    inputs = {}

    inputs[0] = tf.placeholder(tf.float32, [None, cfg.feature_len * att])
    output = tf.placeholder(tf.float32, [None, cfg.num_output])
    for a in range(cfg.feature_len*2):
        inputs[a + 1] = tf.placeholder(tf.float32, [None, att])

    output = tf.placeholder(tf.float32, [None, cfg.num_output])

    input_dic = {}
    for a in range(cfg.feature_len*2+1):
        input_dic['input{}'.format(a)] = inputs[a]

    net = StackNet4(input_dic)
    net.real_setup(cfg.feature_len,
                   cfg.nodes, num_out=cfg.num_output, att=att, verbose=False)

    xy0 = net.layers['output{}'.format(cfg.feature_len)]
    xy1 = net.layers['output{}'.format(cfg.feature_len*2)]

    loss0 = tf.reduce_sum(tf.square(tf.subtract(xy0, output)))
    loss1 = tf.reduce_sum(tf.square(tf.subtract(xy1, output)))
    loss = loss0 # + loss1

    opt = tf.train.AdamOptimizer(learning_rate=cfg.lr, beta1=0.9,
                    beta2=0.999, epsilon=0.00000001,
                    use_locking=False, name='Adam').\
        minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    t00 = datetime.datetime.now()

    with tf.Session() as sess:
        sess.run(init)
        if renetFile:
            saver.restore(sess, renetFile)

        str1 = ''
        for a in range(iterations):

            tr_pre_data = tr.prepare()
            tr_loss, tr_median = run_data_avg4(tr_pre_data, input_dic, sess, [xy0, xy1], 'tr')

            te_pre_data = te.prepare()
            te_loss, te_median = run_data_avg4(te_pre_data, input_dic, sess, [xy0, xy1], 'te')

            t1 = datetime.datetime.now()
            str = "it: {0:.2f} {1:.2f}".format(a*cfg.loop/1000.0,
                                               (t1 - t00).total_seconds()/3600.0)

            for b in range(len(tr_loss)):
                str += " {0:.3f} {1:.3f} {2:.3f} {3:.3f} ".\
                    format(tr_loss[b], te_loss[b], tr_median[b], te_median[b])

            print str, str1

            tl0 = 0
            tl1 = 0
            nt = 0.0
            for _ in range(cfg.loop):
                tr_pre_data = tr.prepare(multi=10)

                while tr_pre_data:
                    for b in tr_pre_data:
                        total_length = len(b[0])
                        sz = b[0].shape[1]/2
                        for c in range(0, total_length, step):
                            feed = {input_dic['input0']: b[0][c:c + cfg.batch_size, :cfg.feature_len*att]}
                            for d in range(cfg.feature_len*2):
                                feed[input_dic['input{}'.format(d + 1)]] = \
                                    b[0][c:c + cfg.batch_size, att * d: att * (d + 1)]
                            feed[output] = b[1][c:c + cfg.batch_size]
                            ll0, ll1, _ = sess.run([loss0, loss1, opt], feed_dict=feed)
                            tl0 += ll0
                            tl1 += ll1
                            nt += len(b[0][c:c + step])
                    tr_pre_data = tr.get_next()

            str1 = "{0:.3f} {1:.3f} ".format(tl0/nt, tl1/nt)
            saver.save(sess, cfg.netFile)

