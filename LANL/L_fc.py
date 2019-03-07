import sys
import tensorflow as tf
import datetime
import numpy as np
import glob
import pickle
import os

sys.path.append('..')
from network import Network
from utils import Utils


def get_dataset(st='../../p_files/L_*.p', avg_file='../../p_files/Avg.p'):

    files = glob.glob(st)
    data = []
    for f in files:
        with open(f, 'r') as fp:
            d = pickle.load(fp)
            data.append(d)

    if not os.path.exists(avg_file):
        d = None
        for dd in data:
            if d is None:
                d = dd[1]
            else:
                d = np.concatenate((d, dd[1]))
        avg = np.mean(d, 0)
        std = np.std(d, 0)
        with open(avg_file, 'w') as fp:
            pickle.dump((avg, std), fp)
    else:
        with open(avg_file, 'r') as fp:
            A = pickle.load(fp)
        avg = A[0]
        std = A[1]

    for n in range(len(data)):
        for a in range(data[n][1].shape[0]):
            data[n][1][a, :] = (data[n][1][a,:]-avg)/std

    return data


def create_data(data, id):
    tr = []
    te = None
    nt = 0
    for d in data:
        if nt == id:
            te = [d]
        else:
            tr.append(d)

        nt += 1

    return te, tr


class sNet3(Network):

    def setup(self):
        pass

    def real_setup(self, nodes, outputs):
        self.feed('data')
        for a in range(len(nodes)):
            name = 'fc_{}'.format(a)
            self.fc(nodes[a], name=name)

        self.fc(outputs, relu=False, name='output')

        print("number of layers = {} {}".format(len(self.layers), nodes))


def run_data(data, inputs, sess, xy):
    results = None
    truth = None

    for b in data:
        feed = {inputs: b[1]}
        result = sess.run(xy, feed_dict=feed)[:, 0]
        if results is None:
            results = result
            truth = b[0]
        else:
            results = np.concatenate((results, result))
            truth = np.concatenate((truth, b[0]))
    return np.mean(np.abs(results-truth))


if __name__ == '__main__':

    data = get_dataset()
    CV = len(data)
    nodes = [124, 32]
    lr = 1e-4
    iterations = 1000
    loop = 10
    batch_size = 100
    netFile = '../../NNs/L_{}'

    for c in range(CV):
        te, tr = create_data(data, c)
        output = tf.placeholder(tf.float32, [None, 1])
        input = tf.placeholder(tf.float32, [None, te[0][1].shape[1]])
        net = sNet3({'data': input})
        net.real_setup(nodes, 1)

        xy = net.layers['output']
        loss = tf.reduce_sum(tf.abs(tf.subtract(xy, output)))

        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                        beta2=0.999, epsilon=0.00000001,
                        use_locking=False, name='Adam').\
            minimize(loss)
        # opt = tf.train.GradientDescentOptimizer(learning_rate=cfg.lr).minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            #if cfg.renetFile:
            #    saver.restore(sess, cfg.renetFile)

            t00 = datetime.datetime.now()
            st1 = ''
            for a in range(iterations):

                total_loss = run_data(tr, input, sess, xy)

                te_loss = run_data(te, input, sess, xy)

                t1 = (datetime.datetime.now()-t00).seconds/3600.0
                str = "it: {0} {1:.3f} {2} {3}".format(
                    a*loop/1000.0, t1, total_loss, te_loss)
                print str, st1

                t_loss = 0
                t_count = 0
                for lp in range(loop):
                    for b in tr:
                        length = len(b[0])
                        b0 = b[0].reshape((length, 1))
                        for d in range(0, length, batch_size):
                            feed = {input: b[1][d:d+batch_size, :],
                                    output: b0[d:d+batch_size, :]}
                            _, A = sess.run([opt, loss], feed_dict=feed)
                            t_loss += A
                            t_count += len(b0[d:d+batch_size])
                    st1 = '{}'.format(t_loss/t_count)

                saver.save(sess, netFile.format(c))


