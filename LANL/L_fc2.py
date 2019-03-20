import sys
import tensorflow as tf
import datetime
import numpy as np
import glob
import pickle
import os
import random
from LANL_Utils import l_utils, sNet3, HOME
from sortedcontainers import SortedList


sys.path.append('{}/my_nets'.format(HOME))
from utils import Utils

#out_location = '/home/weihao/Projects/p_files/L/L_{}'
#netFile = '/home/weihao/Projects/NNs/L/C_{}/L'
#nodes = [1024, 128]


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

def run_data(truth, features, inputs, sess, xy, filename=None):
    #truth, features = l_utils.prepare_data(data, c)

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


def nn_fit2(config, cntn):
    cfg = Utils.load_json_file(config)
    data_file = cfg['eval_file'].format(HOME)
    dd = np.array(Utils.read_csv(data_file)).astype(float)
    print dd.shape

    T = dd[:, 1]
    F = np.concatenate((dd[:,:1], dd[:,3:]), axis=1)

    T1 = []
    T2 = []
    F1 = []
    F2 = []
    for a in range(len(T)):
        if np.random.random()>0.2:
            T1.append(T[a])
            F1.append(F[a,:])
        else:
            T2.append(T[a])
            F2.append(F[a,:])

    att = F.shape[1]
    node2 = map(int, cfg['node2'].split(','))
    netFile = cfg['netFile2']

    # CV = 5
    #nodes = [ 256, 16]
    # nodes = [256, 16]
    lr = 1e-4
    iterations = 1000
    loop = 100
    batch_size = 100

    output = tf.placeholder(tf.float32, [None, 1])
    input = tf.placeholder(tf.float32, [None, att])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    net = sNet3({'data': input})
    net.real_setup(node2, 1, keep_prob=1)

    xy = net.layers['output']
    loss = tf.reduce_sum(tf.abs(tf.subtract(xy, output)))

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
            saver.restore(sess, netFile.format(HOME))

        t00 = datetime.datetime.now()
        st1 = ''
        for a in range(iterations):

            te_loss = run_data(T2, F2, input, sess, xy, '{}/../tmp/te.csv'.format(HOME))
            tr_loss = run_data(T1, F1, input, sess, xy, '{}/../tmp/tr.csv'.format(HOME))

            t1 = (datetime.datetime.now()-t00).seconds/3600.0
            str = "it: {0} {1:.3f} {2} {3} {4}".format(
                a*loop/1000.0, t1, lr, tr_loss, te_loss)
            print str, st1

            t_loss = 0
            t_count = 0
            truth = np.array(T1)
            features = np.array(F1)
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

            saver.save(sess, netFile.format(HOME))

    tf.reset_default_graph()


if __name__ == '__main__':
    config = 'config.json'
    nn_fit2(config, False)