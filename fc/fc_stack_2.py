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
    step = js["step"]
    stage = js["stage"]

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '/fc'

    tr = DataSet(tr_data, batch_size, feature_len + stack)
    te = DataSet(te_data, batch_size, feature_len + stack)

    att = te.sz[1]
    iterations = 10000
    loop = 40
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
    l3 = tf.reduce_sum(tf.square(tf.subtract(xy[0], output)))
    l4 = tf.reduce_sum(tf.square(tf.subtract(xy[0]+xy[1], output)))
    l5 = tf.reduce_sum(tf.square(tf.subtract(xy[0]+xy[1]+xy[2], output)))

    loss = l5 + l4 + l3 #+ l2 + l1 + l0

    opt = None

    if stage<0:
        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                        beta2=0.999, epsilon=0.00000001,
                        use_locking=False, name='Adam').\
            minimize(loss)
    if stage == 0:
        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                        beta2=0.999, epsilon=0.00000001,
                        use_locking=False, name='Adam').\
            minimize(l3)
    if stage == 1:
        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                        beta2=0.999, epsilon=0.00000001,
                        use_locking=False, name='Adam').\
            minimize(l4)
    if stage == 2:
        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                        beta2=0.999, epsilon=0.00000001,
                        use_locking=False, name='Adam').\
            minimize(l5)

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
            tr_loss, tr_median = run_data_stack_avg2(tr_pre_data, input_dic, sess, xy, stack)

            te_pre_data = te.prepare()
            te_loss, te_median = run_data_stack_avg2(te_pre_data, input_dic, sess, xy, stack)

            t1 = datetime.datetime.now()
            str = "it: {0} {1:.3f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}" \
                  " {6:.4f} {7:.4f} {8:.4f} {9:.4f} {10:.4f} {11:.4f} " \
                  "{12:.4f} {13:.4f}".format(
                a*loop/1000.0, (t1 - t00).total_seconds()/3600.0,
                tr_loss[stack-2], tr_loss[stack-1], tr_loss[stack],
                te_loss[stack-2], te_loss[stack-1], te_loss[stack],
                tr_median[stack-2], tr_median[stack-1], tr_median[stack],
                te_median[stack-2], te_median[stack-1], te_median[stack])

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
                            _, ll3,ll4,ll5 = sess.run([opt, l3, l4, l5], feed_dict=feed)
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

