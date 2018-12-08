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

    def real_setup(self, num_out=3, verbose=True):
        # self.parameters(stack, dim_output=num_out)
        self.dim_ref = 64
        self.dim_output = num_out
        ref_out_name = 'fc1'

        # base net
        (self.feed('input').
         fc(2048, name='fc00').
         fc(256, name='fc01').
         fc(self.dim_ref, name=ref_out_name).
         fc(self.dim_output, relu=False, name='output0')
         )

        (self.feed('input', ref_out_name).
         concat(1, name='ic_in').
         fc(2048, name='fc10').
         fc(256, name='fc11').
         fc(self.dim_output, relu=False, name='output1')
         )

        if verbose:
            print("number of layers = {}".format(len(self.layers)))
            for l in sorted(self.layers.keys()):
                print l, self.layers[l].get_shape()

def run_data_avg4(data, inputs, sess, xy, fname):
    rst_dic = {}
    truth_dic = {}
    length = len(xy)
    for b in data:
        feed = {inputs['input']: b[0]}
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

    tr = DataSet(cfg.tr_data, cfg.batch_size, cfg.feature_len)
    te = DataSet(cfg.te_data, cfg.batch_size, cfg.feature_len)
    tr.set_t_scale(cfg.t_scale)
    te.set_t_scale(cfg.t_scale)
    tr.set_num_output(cfg.num_output)
    te.set_num_output(cfg.num_output)
    step = cfg.step

    att = te.sz[1]
    iterations = int(1e6)
    print "input attribute", att, "LR", cfg.lr, 'feature', cfg.feature_len

    input0 = tf.placeholder(tf.float32, [None, cfg.feature_len*att])
    output = tf.placeholder(tf.float32, [None, cfg.num_output])

    input_dic = {'input': input0}

    net = StackNet4(input_dic)
    net.real_setup(num_out=cfg.num_output, verbose=False)

    xy0 = net.layers['output0']
    xy1 = net.layers['output1']

    loss0 = tf.reduce_sum(tf.square(tf.subtract(xy0, output)))
    loss1 = tf.reduce_sum(tf.square(tf.subtract(xy1, output)))
    loss = loss0 + loss1

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
                        for c in range(0, total_length, step):
                            feed = {input0:b[0][c:c + step, :], output:b[1][c:c + step]}
                            ll0, ll1, _ = sess.run([loss0, loss1, opt], feed_dict=feed)
                            tl0 += ll0
                            tl1 += ll1
                            nt += len(b[0][c:c + step])
                    tr_pre_data = tr.get_next()

            str1 = "{0:.3f} {1:.3f} ".format(tl0/nt, tl1/nt)
            saver.save(sess, cfg.netFile)

