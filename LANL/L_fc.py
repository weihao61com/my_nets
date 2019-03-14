import sys
import tensorflow as tf
import datetime
import numpy as np
import glob
import pickle
import os
import random
from LANL_Utils import l_utils, sNet3


def create_data(data, id):
    tr = []
    te = None
    nt = 0
    for d in data:
        if nt == id:
            te = [[data[d][0]]]
        else:
            tr.append(data[d])
        nt += 1

    return te, tr

def run_data(data, c, inputs, sess, xy, filename=None):
    truth, features = l_utils.prepare_data(data, c)

    feed = {inputs: features}
    results = sess.run(xy, feed_dict=feed)[:, 0]

    if filename is not None:
        with open(filename, 'w') as fp:
            skip = int(len(truth)/2000)
            if skip==0:
                skip=1
            for a in range(len(truth)):
                if a%skip==0:
                    fp.write('{},{}\n'.format(results[a], truth[a]))

    return np.mean(np.abs(results-truth))


if __name__ == '__main__':

    locs = ['L_0', 'L_1', 'L_2', 'L_3', 'L_4', 'L_5', 'L_6', 'L_7', 'L_8', 'L_9']
    data = l_utils.get_dataset('/home/weihao/Projects/p_files/L10000', locs)

    CV = 5
    nodes = [1024, 128]
    #nodes = [ 256, 16]
    # nodes = [256, 16]
    lr0 = 1e-5
    iterations = 1000
    loop = 1
    batch_size = 100
    netFile = '../../NNs/L/L_{}'
    att = 250
    cntn = False

    for c in range(CV):
        lr = lr0
        #te, tr = create_data(data, c)
         #len(te[0][0][0][1])
        output = tf.placeholder(tf.float32, [None, 1])
        input = tf.placeholder(tf.float32, [None, att])
        learning_rate = tf.placeholder(tf.float32, shape=[])

        net = sNet3({'data': input})
        net.real_setup(nodes, 1)

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

        with tf.Session() as sess:
            sess.run(init)

            if cntn:
                saver.restore(sess, netFile.format(c))

            t00 = datetime.datetime.now()
            st1 = ''
            for a in range(iterations):

                te_loss = run_data(data[0], c+1, input, sess, xy, '/home/weihao/tmp/te.csv')

                tr_loss = run_data(data[0], -c-1, input, sess, xy, '/home/weihao/tmp/tr.csv')

                t1 = (datetime.datetime.now()-t00).seconds/3600.0
                str = "it: {0} {1:.3f} {2} {3} {4}".format(
                    a*loop/1000.0, t1, lr, tr_loss, te_loss)
                print str, st1

                t_loss = 0
                t_count = 0
                for dd in data:
                    truth, features = l_utils.prepare_data(dd, -c-1, rd=True)
                    length = len(truth)
                    b0 = truth.reshape((length, 1))
                    for lp in range(loop):
                        for d in range(0, length, batch_size):
                            feed = {input: features[d:d+batch_size, :],
                                    output: b0[d:d+batch_size, :],
                                    learning_rate: lr
                            }
                            _, A = sess.run([opt, loss], feed_dict=feed)
                            t_loss += A
                            t_count += len(b0[d:d+batch_size])
                st1 = '{}'.format(t_loss/t_count)

                saver.save(sess, netFile.format(c))
                #lr *= 0.99
            if lr<1e-6:
                break
        break