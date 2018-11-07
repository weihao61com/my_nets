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
        p1 = 'net1'
        p2 = 'net2'
        self.att = att

        self.data_len = att*feature_len
        self.input1 = tf.placeholder(tf.float32, [None, att*feature_len])
        self.output1 = tf.placeholder(tf.float32, [None, num_output])
        #self.reference1 = tf.placeholder(tf.float32, [None, num_ref])

        self.input2 = tf.placeholder(tf.float32, [None, att+num_ref])
        self.output2 = tf.placeholder(tf.float32, [None, num_output])
        #self.reference2 = tf.placeholder(tf.float32, [None, num_ref])

        nodes2.append(num_output)
        self.net1 = P1Net1({'input_{}'.format(p1): self.input1})
        self.net1.real_setup(nodes1, nodes2, p1)

        nodes3.append(num_ref)
        self.net2 = P1Net1({'input_{}'.format(p2): self.input2})
        self.net2.real_setup(nodes3, nodes2, p2)

        self.xy1 = self.net1.layers['output_{}'.format(p1)]
        loss1 = tf.reduce_sum(tf.square(tf.subtract(self.xy1, self.output1)))
        self.opt1 = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                                     beta2=0.999, epsilon=0.00000001,
                                     use_locking=False, name='Adam_{}'.format(p1)). \
            minimize(loss1)
        self.reference1 = self.net1.layers['reference_{}'.format(p1)]

        self.xy2 = self.net2.layers['output_{}'.format(p2)]
        loss2 = tf.reduce_sum(tf.square(tf.subtract(self.xy2, self.output2)))
        self.opt2 = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                                     beta2=0.999, epsilon=0.00000001,
                                     use_locking=False, name='Adam_'.format(p2)). \
            minimize(loss2)
        self.reference2 = self.net2.layers['reference_{}'.format(p2)]

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def train(self, data, truth):
        truth = truth.reshape((1, len(truth)))
        feed = {self.input1: data[:, :self.data_len], self.output1: truth}
        _, reference = sess.run([self.opt1, self.reference1], feed_dict=feed)

        for d in range(self.data_len, len(data), self.att):
            input_data = np.concatenate((reference, data[d:d+self.att]))
            feed = {self.input2: input_data, self.output2: truth}
            _, reference = sess.run([self.opt2, self.reference2], feed_dict=feed)

    def run(self, data):
        # print data.shape
        feed = {net.input1: data[:, :self.data_len]}
        reference, reference1 = sess.run([self.reference1, self.xy1], feed_dict=feed)
        data_length = data.shape[1]
        for d in range(self.data_len, data_length, self.att):
            input_data = np.concatenate((reference, data[:, d:d+self.att]), axis=1)
            feed = {self.input2: input_data}
            if d+self.att == data_length:
                reference = sess.run(self.xy2, feed_dict=feed)
            else:
                reference = sess.run(self.reference2, feed_dict=feed)

        return reference1, reference


def run_data(data, net, sess):
    result1 = None
    results = None
    truth = None

    for b in data:
        for a in range(len(b[0])):
            r1, result = net.run(b[0][a])
            if results is None:
                results = result
                result1 = r1
                truth = b[1][a].reshape((1,len(b[1][a])))
            else:
                result1 = np.concatenate((result1, r1))
                results = np.concatenate((results, result))
                truth = np.concatenate((truth, b[1][a].reshape((1,len(b[1][a])))))

    return Utils.calculate_loss(results, truth), Utils.calculate_loss(result1, truth)


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
    print "input shape", att, "LR", lr, 'feature', feature_len

    net =P1Net(nodes1, nodes2, nodes3, att, num_output, feature_len, lr)

    with tf.Session() as sess:
        sess.run(net.init)
        if renetFile:
            net.saver.restore(sess, renetFile)

        t00 = datetime.datetime.now()

        for a in range(iterations):

            tr_pre_data = tr.prepare_stack()
            X = run_data(tr_pre_data, net, sess)
            tr_loss, tr_median = X[0]
            tr1, tr_med1 = X[1]

            te_pre_data = te.prepare_stack()
            X = run_data(te_pre_data, net, sess)
            te_loss, te_median = X[0]
            te1, te_med1 = X[1]

            t1 = datetime.datetime.now()
            str = "iteration: {} {} {} {} {} {} {} {} {} time {}".format(
                a*loop, tr_loss, te_loss, tr1, te1,
                tr_median, te_median, tr_med1, te_med1, t1 - t00)
            print str
            t00 = t1

            for _ in range(loop):
                tr_pre_data = tr.prepare_stack()
                while tr_pre_data:
                    for b in tr_pre_data:
                        for a in range(len(b[0])):
                            net.train(b[0][a], b[1][a])
                    tr_pre_data = tr.get_next()

            net.saver.save(sess, netFile)

        print netFile
        net.saver.save(sess, netFile)

