import sys
from fc_dataset import *
import tensorflow as tf
import datetime
import os

sys.path.append('..')
from utils import Utils

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

if __name__ == '__main__':

    config_file = "config.json"

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
    batch_size = int(js['batch_size'])
    feature_len = int(js['feature'])
    lr = float(js['lr'])
    step = js["step"]

    num_output = int(js["num_output"])
    nodes = map(int, js["nodes"].split(','))
    nodes.append(num_output)

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '/fc'

    net_type = "fc"
    if 'net_type' in js:
        net_type = js['net_type']

    tr = DataSet(tr_data, batch_size, feature_len)
    te = DataSet(te_data, batch_size, feature_len)
    tr.set_net_type(net_type)
    tr.set_num_output(num_output)
    te.set_net_type(net_type)
    te.set_num_output(num_output)

    sz_in = te.sz
    iterations = 10000
    loop = 10
    t_scale = 10
    if "loop" in js:
        loop = js["loop"]

    print "input shape", sz_in, "LR", lr, 'feature', feature_len

    output = tf.placeholder(tf.float32, [None, num_output])

    if net_type == 'cnn':
        input = tf.placeholder(tf.float32, [None, feature_len, sz_in[1], 1])
        net = cNet({'data': input})
        net.real_setup(nodes)
    else:
        input = tf.placeholder(tf.float32, [None, feature_len * sz_in[1]])
        net = sNet3({'data': input})
        net.real_setup(nodes)

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
        st1 = ''
        for a in range(iterations):

            tr_pre_data = tr.prepare(t_scale=t_scale)
            total_loss, tr_median = run_data(tr_pre_data, input, sess, xy, 'tr')

            te_pre_data = te.prepare(t_scale=t_scale)
            te_loss, te_median = run_data(te_pre_data, input, sess, xy, 'te')

            t1 = (datetime.datetime.now()-t00).seconds/3600.0
            str = "iteration: {0} {1:.3f} {2} {3} {4} {5}".format(
                a*loop/1000.0, t1, total_loss, te_loss,
                tr_median, te_median)
            print str, st1

            t_loss = 0
            t_count = 0
            for lp in range(loop):
                tr_pre_data = tr.prepare(rdd=True, multi=1, t_scale=t_scale)
                while tr_pre_data:
                    for b in tr_pre_data:
                        length = len(b[0])
                        for c in range(0, length, step):
                            feed = {input: b[0][c:c+step], output: b[1][c:c+step]}
                            _, A = sess.run([opt, loss], feed_dict=feed)
                            t_loss += A
                            t_count += len(b[0][c:c+step])
                    tr_pre_data = tr.get_next()
                st1 = '{}'.format(t_loss/t_count)

            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)

