import matplotlib.pyplot as plt
import sys
import glob
import numpy as np
import pickle
import os
from LANL_Utils import l_utils, sNet3
import tensorflow as tf

SEGMENT = 150000


SEG = 10000
CV = 5
dct = False
dim = 200
threads = 2
location = '/home/weihao/tmp/L'
out_location = '/home/weihao/Projects/p_files/L/L_{}'
nodes = [1024, 128]
eval_file = '/home/weihao/tmp/fit.csv'
step = 1000
att = dim+1


def get_values(lines):
    x = []
    y = []
    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])

    return np.mean(np.array(y)), x


fp0 = open(eval_file, 'w')

for c in range(CV):

    netFile = '/home/weihao/Projects/NNs/L/L_{}'.format(c)
    files = glob.glob(os.path.join(location, 'L_*.csv'))
    idx = l_utils.rdm_ids(files)

    output = tf.placeholder(tf.float32, [None, 1])
    input = tf.placeholder(tf.float32, [None, att])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    avg_file = os.path.join('/home/weihao/Projects/p_files/L', 'Avg.p')
    with open(avg_file, 'r') as fp:
        A = pickle.load(fp)
    avgf = A[0]
    stdf = A[1]
    avg0 = A[2]

    net = sNet3({'data': input})
    net.real_setup(nodes, 1)
    xy = net.layers['output']

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        saver.restore(sess, netFile)
        for filename in idx:
            if not idx[filename] == c:
                continue

            #filename = '/home/weihao/tmp/L/L_11.csv'

            with open(filename, 'r') as fp:
                line0 = fp.readlines()

                print len(line0)
                start = 0
                seg_step = 1000000

                for start in range(0, len(line0), seg_step):

                    lines = line0[start:start+SEGMENT]
                    avg, x = get_values(lines)

                    a = 0
                    r = []
                    while a<=len(x)-SEG:

                        features = l_utils.feature_final(x[a:a+SEG], dct, dim)
                        features = (features-avgf)/stdf
                        features = features.reshape((1, len(features)))
                        feed = {input: features}
                        results = sess.run(xy, feed_dict=feed)[:, 0]
                        a += step
                        r.append(results[0])

                    fp0.write('{},{},{},{},{}\n'.format(c, avg, len(r), np.mean(r)+avg0, np.median(r)+avg0))
                    # print c, avg, len(r), np.mean(r)+avg0, np.median(r)+avg0
    tf.reset_default_graph()

fp0.close()
