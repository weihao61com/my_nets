import sys
from fc_dataset import *
import tensorflow as tf
import datetime

HOME = '{}/Projects/'.format(os.getenv('HOME'))
sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils
from fc_dataset import DataSet

if __name__ == '__main__':

    config_file = "config_pyra.json"

    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    js = Utils.load_json_file(config_file)

    tr_data = []
    te_data = []
    for key in js:
        if key.startswith('tr'):
            tr_data.append(HOME + js[key])
        if key.startswith('te'):
            te_data.append(HOME + js['te'])

    netFile = HOME + 'NNs/' + js['net'] + '/fc'

    batch_size = js['batch_size']
    feature_len = js['feature']
    lr = js['lr']
    base = js['base']
    step = js["step"]
    num_output = 3

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '/fc'

    tr = DataSet(tr_data, batch_size, feature_len * base)
    te = DataSet(te_data, batch_size, feature_len * base)

    att = te.sz[1]
    iterations = 10000
    loop = 10
    print "input attribute", att, "LR", lr, 'feature', feature_len

    inputs = {}
    input_dic = {}

    output = tf.placeholder(tf.float32, [None, num_output])
    for a in range(base):
        inputs[a] = tf.placeholder(tf.float32, [None, att*feature_len])
        input_dic['input{}'.format(a)] = inputs[a]

    net = PyraNet(input_dic)
    net.real_setup(base)

    xy = {}
    for a in range(base + 1):
        xy[a] = net.layers['output{}'.format(a)]

    ls = []
    loss = None
    for x in range(base):
        ll = tf.reduce_sum(tf.square(tf.subtract(xy[x], output)))
        if loss is None:
            loss = ll
        else:
            loss = loss + ll
        ls.append(ll)

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                                 beta2=0.999, epsilon=0.00000001,
                                 use_locking=False, name='Adam'). \
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
            tr_loss, tr_median = run_data_base(tr_pre_data, input_dic, sess, xy, base)

            te_pre_data = te.prepare()
            te_loss, te_median = run_data_base(te_pre_data, input_dic, sess, xy, base)

            t1 = datetime.datetime.now()
            str = "it: {0} {1:.2f}".format(a * loop / 1000.0, (t1 - t00).total_seconds() / 3600.0)
            for s in range(base + 1):
                str += " {0:.3f} {1:.3f} {2:.3f} {3:.3f}".format(tr_loss[s], te_loss[s], tr_median[s], te_median[s])

            print str, str1

            tl3 = 0
            tl4 = 0
            tl5 = 0
            nt = 0
            for _ in range(loop):
                tr_pre_data = tr.prepare(multi=50)

                while tr_pre_data:
                    for b in tr_pre_data:
                        total_length = len(b[0])
                        for c in range(0, total_length, step):
                            length = b[0].shape[1] - att * stack
                            feed = {input_dic['input0']: b[0][c:c + step, :length]}
                            for d in range(stack):
                                feed[input_dic['input{}'.format(d + 1)]] = \
                                    b[0][c:c + step, length + 4 * d:length + 4 * (d + 1)]
                            feed[output] = b[1][c:c + step]
                            _, ll3, ll4, ll5 = sess.run([opt, ls[0], ls[1], ls[-1]], feed_dict=feed)
                            tl3 += ll3
                            tl4 += ll4
                            tl5 += ll5
                            nt += len(b[0][c:c + step])
                    tr_pre_data = tr.get_next()

                    tr_pre_data = tr.get_next()
            str1 = "{0:.4f} {1:.4f} {2:.4f}".format(tl3 / nt, tl4 / nt, tl5 / nt)
            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)
