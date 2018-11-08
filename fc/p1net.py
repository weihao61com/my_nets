import sys
from fc_dataset import DataSet, P1Net1
import tensorflow as tf
import datetime
import os
import numpy as np

sys.path.append('..')
from utils import Utils

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'


class P1Net:
    def __init__(self, nodes1, nodes2, nodes3, att, num_output, feature_len, lr):
        num_ref = nodes1[-1]
        self.p1 = 'net1'
        self.p2 = 'net2'
        self.att = att
        self.step = 0
        self.lr = lr

        self.data_len = att*feature_len
        self.input1 = tf.placeholder(tf.float32, [None, att*feature_len])
        self.output1 = tf.placeholder(tf.float32, [None, num_output])
        #self.reference1 = tf.placeholder(tf.float32, [None, num_ref])

        self.input2 = tf.placeholder(tf.float32, [None, att+num_ref])
        self.output2 = tf.placeholder(tf.float32, [None, num_output])
        #self.reference2 = tf.placeholder(tf.float32, [None, num_ref])
        self.learning_rate = tf.placeholder(tf.float32)

        nodes2.append(num_output)
        self.net1 = P1Net1({'input_{}'.format(self.p1): self.input1})
        self.net1.real_setup(nodes1, nodes2, self.p1)

        nodes3.append(num_ref)
        self.net2 = P1Net1({'input_{}'.format(self.p2): self.input2})
        self.net2.real_setup(nodes3, nodes2, self.p2)

        self.xy1 = self.net1.layers['output_{}'.format(self.p1)]
        self.loss1 = tf.reduce_sum(tf.square(tf.subtract(self.xy1, self.output1)))

        self.reference1 = self.net1.layers['reference_{}'.format(self.p1)]

        self.xy2 = self.net2.layers['output_{}'.format(self.p2)]
        self.loss2 = tf.reduce_sum(tf.square(tf.subtract(self.xy2, self.output2)))
        self.reference2 = self.net2.layers['reference_{}'.format(self.p2)]

        self.saver = tf.train.Saver()

        self.opt1 = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                                           beta2=0.999, epsilon=0.00000001,
                                           use_locking=False, name='Adam_{}'.format(self.p1)). \
            minimize(self.loss1)
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,
                                     beta2=0.999, epsilon=0.00000001,
                                     use_locking=False, name='Adam_'.format(self.p2)). \
            minimize(self.loss2)
        self.init = tf.global_variables_initializer()

    def train(self, data):
        truth = []
        input_data = []
        for d in data:
            truth.append(d[2])
            input_data.append(d[0])

        truth = np.array(truth)
        input_data = np.array(input_data)

        feed = {self.input1: input_data, self.output1: truth}

        if step==0:
            _, loss = sess.run([self.opt1, self.loss1], feed_dict=feed)
            return loss

        loss = 0
        reference = sess.run(self.reference1, feed_dict=feed)
        for a in range(len(data)):
            ref = reference[a]
            for d in data[a][1]:
                input_data = np.concatenate((ref, d))
                input_data = input_data.reshape((1, len(input_data)))
                tr = truth[a]
                tr = tr.reshape(1, len(tr))
                feed = {self.input2: input_data, self.output2: tr,
                        self.learning_rate:self.lr/1000}
                ref, _ = sess.run([self.reference2, self.opt2], feed_dict=feed)
                ref = ref[0]
            loss += sess.run(self.loss2, feed_dict=feed)

        return loss

    def run(self, data, sess):
        input_data = []
        for d in data:
            input_data.append(d[0])
        input_data = np.array(input_data)
        feed = {net.input1: np.array(input_data)}
        reference, results1 = sess.run([self.reference1, self.xy1], feed_dict=feed)

        if self.step==0:
            results2 = results1
            return results1, results2

        results2 = []
        for a in range(len(data)):
            ref = reference[a]
            for d in data[a][1]:
                input_data = np.concatenate((ref, d))
                input_data= input_data.reshape((1, len(input_data)))
                feed = {self.input2: input_data}
                ref = sess.run(self.reference2, feed_dict=feed)[0]

            rst = sess.run(self.xy2, feed_dict=feed)[0]
            results2.append(rst)

        return results1, np.array(results2)

    def run_data(self, data, sess):
        result1 = None
        result2 = None
        truth = None

        for b in data:
            r1, r2 = self.run(b, sess)
            t0 = None
            for a in range(len(b)):
                t = b[a][2].reshape((1,len(b[a][2])))
                if t0 is None:
                    t0 = t
                else:
                    t0 = np.concatenate((t0, t))
            if result1 is None:
                result1 = r1
                result2 = r2
                truth = t0
            else:
                truth = np.concatenate((truth, t0))
                result2 = np.concatenate((result2, r2))
                result1 = np.concatenate((result1, r1))

        return Utils.calculate_loss(result1, truth), Utils.calculate_loss(result2, truth)


if __name__ == '__main__':

    config_file = "{}/my_nets/fc/config_p1.json".format(HOME)

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    js = Utils.load_json_file(config_file)

    tr_data = []
    te_data = []
    for key in js:
        if key.startswith('tr'):
            tr_data.append(HOME + js[key])
        if key.startswith('te'):
            te_data.append(HOME + js['te'])

    netFile = HOME + 'NNs/' + js['net'] + '/p1'
    batch_size = int(js['batch_size'])
    feature_len = int(js['feature'])
    lr = float(js['lr'])
    step = js['step']

    num_output = 3
    nodes1 = map(int, js["nodes1"].split(','))
    nodes2 = map(int, js["nodes2"].split(','))
    nodes3 = map(int, js["nodes3"].split(','))

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '/p1'

    tr = DataSet(tr_data, batch_size, feature_len+1)
    te = DataSet(te_data, batch_size, feature_len+1)

    att = te.sz[1]
    iterations = 100000
    loop = 2
    if step==0:
        loop=200
    print "input shape", att, "LR", lr, 'feature', feature_len

    net = P1Net(nodes1, nodes2, nodes3, att, num_output, feature_len, lr)
    net.step = step

    with tf.Session() as sess:
        sess.run(net.init)
        if renetFile:
            net.saver.restore(sess, renetFile)

        t00 = datetime.datetime.now()
        st1 = ''
        for a in range(iterations):

            tr_pre_data = tr.prepare_stack()
            X = net.run_data(tr_pre_data, sess)
            tr_loss, tr_median = X[0]
            tr1, tr_med1 = X[1]

            te_pre_data = te.prepare_stack()
            X = net.run_data(te_pre_data, sess)
            te_loss, te_median = X[0]
            te1, te_med1 = X[1]

            t1 = (datetime.datetime.now()-t00).seconds
            str = "iteration: {0} {1:.3f} {2:.3f} {3:.3f} {4:.3f} " \
                  "{5:.3f} {6:.3f} {7:.3f} {8:.3f} {9}".format(
                a*loop/1000.0, tr_loss, te_loss, tr1, te1,
                tr_median, te_median, tr_med1, te_med1, t1)
            print str, st1
            #t00 = datetime.datetime.now()

            tl = 0
            tn = 0
            for _ in range(loop):
                tr_pre_data = tr.prepare_stack()
                while tr_pre_data:
                    for b in tr_pre_data:
                        loss = net.train(b)
                        tl += loss
                        tn += len(b)
                    tr_pre_data = tr.get_next()

            net.saver.save(sess, netFile)
            st1 = '{}'.format(tl/tn)
        print netFile
        net.saver.save(sess, netFile)

