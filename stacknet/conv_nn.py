
import tensorflow as tf
import sys
from o2_load import *

HOME = '/home/weihao/Projects/posenet/'
sys.path.append(HOME + 'my_nets/paranet')
from network import Network
import time
from simple_fc_dataset import *


if __name__ == '__main__':

    tr_data = HOME + 'my_nets/stacknet/c2_tr.p'
    te_data = HOME + 'my_nets/stacknet/c2_te.p'
    netFile = HOME + '/cNet/output_fc'
    batch_size = 500

    re_train = False

    tr = DataSet(load_data(tr_data), batch_size)

    te_set = DataSet(load_data(te_data), batch_size)

    sz_in = te_set.sz
    lr = 1e-4
    iterations = 1000
    loop = 500
    print "input shape", sz_in, "LR", lr

    input = tf.placeholder(tf.float32, [None, sz_in[0], sz_in[1], 1])
    output = tf.placeholder(tf.float32, [None, 2])

    net = cNet({'data': input})
    xy = net.layers['output']
    # loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy, output))))
    loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))

    #opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                        beta2=0.999, epsilon=0.00000001,
                        use_locking=False, name='Adam').\
        minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        if re_train:
            saver.restore(sess, netFile)

        t00 = datetime.datetime.now()

        for a in range(iterations):

            total_loss = 0
            tr_pre_data = tr.prepare2()
            for b in tr_pre_data:
                feed = {input: b[0], output: b[1]}
                ll = sess.run(loss, feed_dict=feed)
                total_loss += ll
            total_loss /= tr.length
            te_loss = 0

            te_pre_data = te_set.prepare2()
            for b in te_pre_data:
                feed = {input: b[0], output: b[1]}
                ll = sess.run(loss, feed_dict=feed)
                te_loss += ll
            te_loss /= te_set.length
            t1 = datetime.datetime.now()
            str = "iteration: {} {} {} {} time {}". \
                  format(a, total_loss, te_loss, te_loss-total_loss, t1 - t00)
            print str
            t00 = t1

            for _ in range(loop):
                tr_pre_data = tr.prepare2()
                for b in tr_pre_data:
                    feed = {input: b[0], output: b[1]}
                    sess.run(opt, feed_dict=feed)

            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)
