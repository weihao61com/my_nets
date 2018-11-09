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

    att = te_set.sz[1]
    iterations = 10000
    loop = 100
    print "input attribute", att, "LR", lr, 'feature', feature_len

    inputs = {}

    inputs[0] = tf.placeholder(tf.float32, [None, feature_len*att])
    output = tf.placeholder(tf.float32, [None, num_output])
    for a in range(stack):
        inputs[a+1] = tf.placeholder(tf.float32, [None, att])

    input_dic = {}
    for a in range(stack+1):
        input_dic['input{}'.format(a)] = inputs[a]

    net = StackNet(input_dic)
    net.real_setup(stack, verbose=False)

    xy = {}
    for a in range(stack+1):
        xy[a] = net.layers['output{}'.format(a)]

    #loss = tf.reduce_sum(tf.square(tf.square(tf.subtract(xy, output))))
    #loss = tf.reduce_sum(tf.square(tf.subtract(xy[0], output)))
    #for a in range(stack):
    #    loss = tf.add(loss, tf.reduce_sum(tf.square(tf.subtract(xy[a+1], output))))

    #l0 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[stack-7], output)))) * .5
    #l1 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[stack-6], output)))) * .6
    #l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[stack-5], output)))) * .7
    l3 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[stack-2], output)))) * .8
    l4 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[stack-1], output)))) * .9
    l5 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy[stack], output))))

    loss = l5 + l4 + l3 #+ l2 + l1 + l0

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
            tr_loss, tr_median = run_data_stack_avg(tr_pre_data, input_dic, sess, xy, stack)

            te_pre_data = te_set.prepare()
            te_loss, te_median = run_data_stack_avg(te_pre_data, input_dic, sess, xy, stack)

            t1 = datetime.datetime.now()
            str = "it: {0} {1:.1f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}" \
                  " {6:.4f} {7:.4f} {8:.4f} {9:.4f} {10:.4f} {11:.4f} " \
                  "{12:.4f} {13:.4f}".format(
                a*loop/1000.0, (t1 - t00).total_seconds(),
                tr_loss[stack-2], tr_loss[stack-1], tr_loss[stack],
                te_loss[stack-2], te_loss[stack-1], te_loss[stack],
                tr_median[stack-2], tr_median[stack-1], tr_median[stack],
                te_median[stack-2], te_median[stack-1], te_median[stack])

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

