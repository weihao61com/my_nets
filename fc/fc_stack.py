import sys
from fc_dataset import *
import tensorflow as tf
import datetime

sys.path.append('..')
from utils import Utils

HOME = '/Users/weihao/Projects/'

if __name__ == '__main__':

    config_file = "config_stack.json"

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

    netFile = HOME + 'NNs/' + js['net'] + '/fc'

    batch_size = js['batch_size']
    feature_len = js['feature']
    lr = js['lr']
    stack = js['stack']
    num_output = js["num_output"]

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '/fc'

    tr = DataSet(tr_data, batch_size, feature_len+stack)
    te_set = DataSet(te_data, batch_size, feature_len+stack)

    sz_in = te_set.sz
    iterations = 10000
    loop = 5
    print "input shape", sz_in, "LR", lr, 'feature', feature_len

    inputs = {}

    inputs[0] = tf.placeholder(tf.float32, [None, feature_len* sz_in[1]])
    output = tf.placeholder(tf.float32, [None, num_output])
    for a in range(stack):
        inputs[a+1] = tf.placeholder(tf.float32, [None, sz_in[1]])

    input_dic = {}
    for a in range(stack+1):
        input_dic['input{}'.format(a)] = inputs[a]

    net = StackNet(input_dic)
    net.real_setup(stack)

    xy = {}
    for a in range(stack):
        xy[a] = net.layers['output{}'.format(a+1)]

    #loss = tf.reduce_sum(tf.square(tf.square(tf.subtract(xy, output))))
    #loss = tf.reduce_sum(tf.square(tf.subtract(xy[0], output)))
    #for a in range(stack):
    #    loss = tf.add(loss, tf.reduce_sum(tf.square(tf.subtract(xy[a+1], output))))

    #l0 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[0], output)))) * 1
    #l1 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[1], output)))) * 2
    #l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[2], output)))) * 3
    #l3 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[3], output)))) * 4
    #l4 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[4], output)))) * 5
    l5 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[stack-1], output))))

    loss = l5

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

            tr_pre_data = tr.prepare()
            total_loss, tr_median = run_data_stack(tr_pre_data, input_dic, sess, xy, stack)

            te_pre_data = te_set.prepare()
            te_loss, te_median = run_data_stack(te_pre_data, input_dic, sess, xy, stack)

            t1 = datetime.datetime.now()
            str = "iteration: {0} {1:.1f} {2:.1f} {3:.1f} {4:.1f} {5:.2f}" \
                  " {6:.2f} {7:.2f} {8:.2f} time {9}".format(
                a*loop, total_loss[stack-2], te_loss[stack-2],total_loss[stack-1], te_loss[stack-1],
                tr_median[stack-2], te_median[stack-2],
                tr_median[stack-1], te_median[stack-1], t1 - t00)

            print str
            t00 = t1

            for _ in range(loop):
                tr_pre_data = tr.prepare() #.get()
                while tr_pre_data:
                    for b in tr_pre_data:
                        length = b[0].shape[1] - 4 * stack
                        feed = {input_dic['input0']: b[0][:, :length]}
                        for a in range(stack):
                            feed[input_dic['input{}'.format(a + 1)]] = \
                                b[0][:, length + 4 * a:length + 4 * (a + 1)]
                        feed[output] = b[1]
                        sess.run(opt, feed_dict=feed)
                    tr_pre_data = tr.get_next()

            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)

