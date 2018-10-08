# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
import cv2
import datetime
import sys
from o2_load import *
import os
import Queue

sys.path.append('/home/weihao/posenet/my_nets/paranet')
from network import Network
import time

class DataSet:
    def __init__(self, data, batch_size=500):
        self.data = data
        self.batch_size = batch_size

    def get_data(self):
        data_gen = self.gen_data()
        sz_in = self.data[0][0].shape
        pre_data = []

        while True:
            inputs = []
            outputs = []
            done = False
            for _ in range(self.batch_size):
                input_p, output_p = next(data_gen, (None, _))
                if input_p is None:
                    done = True
                    break
                inputs.append(input_p.reshape(sz_in[0] * sz_in[1]))
                outputs.append(output_p.reshape(2))

            if len(inputs) > 0:
                pre_data.append((inputs, outputs))
            if done:
                break

        return pre_data

    def gen_data(self):
        indices = range(len(self.data))
        np.random.shuffle(indices)
        for a in indices:
            input = self.data[a][0]
            np.random.shuffle(input)
            output = self.data[a][1][:2]
            yield input, output

class sNet(Network):

    def setup(self):
        (self.feed('data').
         fc(200, name='fc1').
         fc(10, name='fc2').
         fc(2, relu=False, name='output'))

        print("number of layers = {}".format(len(self.layers)))

def fill_gueue(tr, pool):
    queue = Queue.Queue()
    for _ in range(pool):
        queue.put(tr.get_data())

    return queue


if __name__ == '__main__':

    # def main():
    tr_data = '/home/weihao/posenet/my_nets/stacknet/c2_tr.p'
    te_data = '/home/weihao/posenet/my_nets/stacknet/c2_te.p'
    netFile = "/home/weihao/posenet/Net/output_fc"
    renetFile = "/home/weihao/posenet/Net/output_fc"
    batch_size = 500

    re_train = False

    tr = DataSet(load_data(tr_data, 50), batch_size)
    te_set = DataSet(load_data(te_data), batch_size).get_data()

    pool = 15
    queue = Queue.Queue()

    len_in = len(te_set[0][0][0])
    len_out = len(te_set[0][1][0])
    lr = 1e-6
    iterations = 10000
    loop = 3000

    input = tf.placeholder(tf.float32, [None, len_in])
    output = tf.placeholder(tf.float32, [None, len_out])

    net = sNet({'data': input})
    xy = net.layers['output']

    # loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy, output))))
    loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    # opt = AdamOptimizer(learning_rate=lr, beta1=0.9,
    #                   beta2=0.999, epsilon=0.00000001,
    #                   use_locking=False, name='Adam').\
    #     minimize(loss)

    # opt = tf.\
    #     train.\
    #     MomentumOptimizer(learning_rate=0.1, momentum=0.9).\
    #     minimize(loss)

    # config = tf.ConfigProto(
    #     inter_op_parallelism_threads=4,
    #     intra_op_parallelism_threads=4
    # )

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # with tf.Session(config=config) as sess:

    with tf.Session() as sess:
        sess.run(init)
        if re_train:
            saver.restore(sess, renetFile)

        t00 = datetime.datetime.now()

        for a in range(iterations):
            for _ in range(loop):
                if queue.empty():
                    queue = fill_gueue(tr, pool)
                tr_set = queue.get()
                for d in tr_set:
                    sess.run(opt,
                             feed_dict={input: d[0], output: d[1]})

            #if queue.empty():
            #    queue = fill_gueue(tr, pool)
            #tr_set = queue.get()
            total_loss = 0
            nt = 0
            for d in tr_set:
                feed = {input: d[0], output: d[1]}
                ll = sess.run(loss, feed_dict=feed)
                total_loss += ll
                nt += len(d[0])
            total_loss /= nt

            te_loss = 0
            nt = 0
            for d in te_set:
                feed = {input: d[0], output: d[1]}
                ll = sess.run(loss, feed_dict=feed)
                te_loss += ll
                nt += len(d[0])
            te_loss /= nt
            t1 = datetime.datetime.now()
            print("iteration: {} {} {} time {}".
                  format(a, total_loss, te_loss, t1 - t00))
            t00 = t1
            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)

#    main()
