import sys
from fc_dataset import DataSet, P1Net
import tensorflow as tf
import datetime
import os
import numpy as np

sys.path.append('..')
from utils import Utils

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'


def run_data(data, inputs, sess, xy, att):
    results = None
    truth = None

    for b in data:
        for a in range(len(b[0])):
            feed = {inputs: b[0][a][:, :att]}
            result = sess.run(xy, feed_dict=feed)
            if results is None:
                results = result
                truth = b[1][a].reshape((1,len(b[1][a])))
            else:
                results = np.concatenate((results, result))
                truth = np.concatenate((truth, b[1][a].reshape((1,len(b[1][a])))))

    return Utils.calculate_loss(results, truth)


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
    nodes2.append(num_output)

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '/p1'

    tr = DataSet(tr_data, batch_size, feature_len+1)
    te = DataSet(te_data, batch_size, feature_len+1)

    att = te.sz[1]
    iterations = 100000
    loop = 10
    print "input shape", att, "LR", lr, 'feature', feature_len
    att *= feature_len

    input = tf.placeholder(tf.float32, [None, att])
    output = tf.placeholder(tf.float32, [None, num_output])

    net =P1Net({'input': input})
    net.real_setup(nodes1, nodes2)

    xy = net.layers['output']
    loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                        beta2=0.999, epsilon=0.00000001,
                        use_locking=False, name='Adam').\
        minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        if renetFile:
            saver.restore(sess, renetFile)

        t00 = datetime.datetime.now()

        for a in range(iterations):

            tr_pre_data = tr.prepare_stack()
            total_loss, tr_median = run_data(tr_pre_data, input, sess, xy, att)

            te_pre_data = te.prepare_stack()
            te_loss, te_median = run_data(te_pre_data, input, sess, xy, att)

            t1 = datetime.datetime.now()
            str = "iteration: {} {} {} {} {} time {}".format(
                a*loop, total_loss, te_loss,
                tr_median, te_median, t1 - t00)
            print str
            t00 = t1

            for _ in range(loop):
                tr_pre_data = tr.prepare_stack()
                while tr_pre_data:
                    for b in tr_pre_data:
                        for a in range(len(b[0])):
                            feed = {input: b[0][a][:, :att],
                                    output: b[1][a].reshape((1,len(b[1][a])))}
                            sess.run(opt, feed_dict=feed)
                    tr_pre_data = tr.get_next()

            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)
