
import tensorflow as tf
import sys
from o2_load import *

HOME = '/home/weihao/Projects/posenet/'
sys.path.append(HOME + 'my_nets/paranet')
from network import Network
import time
from simple_fc_dataset import *


if __name__ == '__main__':
    loop = int(sys.argv[1])

    te_data = HOME + 'my_nets/stacknet/c2_te.p'
    netFile = HOME + '/cNet/output_fc'
    batch_size = 500

    te_set = DataSet(load_data(te_data), batch_size)

    sz_in = te_set.sz

    input = tf.placeholder(tf.float32, [None, sz_in[0], sz_in[1], 1])
    output = tf.placeholder(tf.float32, [None, 2])

    net = cNet({'data': input})
    xy = net.layers['output']
    # loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xy, output))))
    loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, netFile)

        rst = {}
        for _ in range(loop):
            te_pre_data = te_set.prepare2(False)
            b = te_pre_data[0]
            feed = {input: b[0]}
            result = sess.run(xy, feed_dict=feed)
            for a in range(len(result)):
                if not a in rst:
                    rst[a] = []
                rst[a].append(result[a])

        truth = b[1]

        d = []
        for a in range(len(truth)):
            r = cal_diff(truth[a], rst[a])
            d.append(r)
        print np.mean(d)


