import sys
from fc_dataset import *
import tensorflow as tf
import datetime
import os

sys.path.append('..')
from utils import Utils

if __name__ == '__main__':

    config_file = "config.json"

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    cfg = Config(config_file)

    tr = DataSet(cfg.tr_data, cfg)
    te = DataSet(cfg.te_data, cfg)

    cfg.att = te.sz[1]
    iterations = 100000

    print "input shape", cfg.att, "LR", cfg.lr, 'feature', cfg.feature_len

    output = tf.placeholder(tf.float32, [None, cfg.num_output])

    if cfg.net_type == 'cnn':
        input = tf.placeholder(tf.float32, [None, cfg.feature_len, cfg.att, 1])
        net = cNet({'data': input})
    else:
        input = tf.placeholder(tf.float32, [None, cfg.feature_len * cfg.att])
        net = sNet3({'data': input})

    net.real_setup(cfg.nodes[0], cfg.num_output)

    xy = net.layers['output']
    loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))

    opt = tf.train.AdamOptimizer(learning_rate=cfg.lr, beta1=0.9,
                        beta2=0.999, epsilon=0.00000001,
                        use_locking=False, name='Adam').\
        minimize(loss)
    # opt = tf.train.GradientDescentOptimizer(learning_rate=cfg.lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        if cfg.renetFile:
            saver.restore(sess, cfg.renetFile)

        t00 = datetime.datetime.now()
        st1 = ''
        for a in range(iterations):

            tr_pre_data = tr.prepare()
            total_loss, tr_median = run_data(tr_pre_data, input, sess, xy, 'tr')

            te_pre_data = te.prepare()
            te_loss, te_median = run_data(te_pre_data, input, sess, xy, 'te')

            t1 = (datetime.datetime.now()-t00).seconds/3600.0
            str = "iteration: {0} {1:.3f} {2} {3} {4} {5}".format(
                a*cfg.loop/1000.0, t1, total_loss, te_loss,
                tr_median, te_median)
            print str, st1

            t_loss = 0
            t_count = 0
            for lp in range(cfg.loop):
                tr_pre_data = tr.prepare(rdd=True, multi=1)
                while tr_pre_data:
                    for b in tr_pre_data:
                        length = len(b[0])
                        for c in range(0, length, cfg.memory_size):
                            feed = {input: b[0][c:c+cfg.memory_size], output: b[1][c:c+cfg.memory_size]}
                            _, A = sess.run([opt, loss], feed_dict=feed)
                            t_loss += A
                            t_count += len(b[0][c:c+cfg.memory_size])
                    tr_pre_data = tr.get_next()
                st1 = '{}'.format(t_loss/t_count)

            saver.save(sess, cfg.netFile)


