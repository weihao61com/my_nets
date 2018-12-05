import sys
from fc_dataset import *
import tensorflow as tf
import datetime
import fc_const

from utils import Utils

if __name__ == '__main__':

    config_file = "config_stack_3.json"

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    cfg = Config(config_file)

    tr = DataSet(cfg.tr_data, cfg.batch_size, cfg.feature_len)
    te = DataSet(cfg.te_data, cfg.batch_size, cfg.feature_len)
    tr.set_t_scale(cfg.t_scale)
    te.set_t_scale(cfg.t_scale)
    tr.set_num_output(cfg.num_output)
    te.set_num_output(cfg.num_output)

    att = te.sz[1]
    iterations = 10000
    print "input attribute", att, "LR", cfg.lr, 'feature', cfg.feature_len

    inputs = {}

    inputs[0] = tf.placeholder(tf.float32, [None, cfg.feature_len*att])
    output = tf.placeholder(tf.float32, [None, cfg.num_output])
    for a in range(cfg.feature_len):
        inputs[a+1] = tf.placeholder(tf.float32, [None, att])

    input_dic = {}
    for a in range(cfg.feature_len+1):
        input_dic['input{}'.format(a)] = inputs[a]

    net = StackNet(input_dic)
    net.real_setup(cfg.feature_len, num_out=cfg.num_output, verbose=False)

    xy = {}
    for a in range(cfg.feature_len+1):
        xy[a] = net.layers['output{}'.format(a)]

    ls = []
    loss = None
    for x in range(feature_len+1):
        ll = tf.reduce_sum(tf.square(tf.subtract(xy[x], output)))
        #ll = tf.reduce_sum(tf.abs(tf.subtract(xy[x], output)))
        if loss is None:
            loss = ll
        else:
            loss = loss + ll
        ls.append(ll)

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
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
            tr_loss, tr_median = run_data_stack_avg3(tr_pre_data, input_dic, sess, xy, 'tr')

            te_pre_data = te.prepare()
            te_loss, te_median = run_data_stack_avg3(te_pre_data, input_dic, sess, xy, 'te')

            t1 = datetime.datetime.now()
            str = "it: {0:.2f} {1:.2f}".format(a*loop/1000.0, (t1 - t00).total_seconds()/3600.0)
            for s in range(0, feature_len+1, 5  ):
                str += " {0:.5f} {1:.5f} {2:.5f} {3:.5f} ".format(tr_loss[s], te_loss[s], tr_median[s], te_median[s])

            print str, str1

            tl3 = 0
            tl4 = 0
            tl5 = 0
            nt = 0
            for _ in range(loop):
                tr_pre_data = tr.prepare(multi=10)

                while tr_pre_data:
                    for b in tr_pre_data:
                        total_length = len(b[0])
                        for c in range(0, total_length, step):
                            length = b[0].shape[1] - att * feature_len
                            feed = {input_dic['input0']: b[0][c:c + step, :]}
                            for d in range(feature_len):
                                feed[input_dic['input{}'.format(d + 1)]] = \
                                    b[0][c:c + step,  4 * d: 4 * (d + 1)]
                            feed[output] = b[1][c:c + step]
                            _, ll3,ll4,ll5 = sess.run([opt, ls[0], ls[1], ls[-1]], feed_dict=feed)
                            tl3 += ll3
                            tl4 += ll4
                            tl5 += ll5
                            nt += len(b[0][c:c + step])
                    tr_pre_data = tr.get_next()

                    tr_pre_data = tr.get_next()
            str1 = "{0:.4f} {1:.4f} {2:.4f}".format(tl3/nt, tl4/nt, tl5/nt)
            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)

