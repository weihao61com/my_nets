import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils, sNet3
import tensorflow as tf

SEG = 150000
seg =  10000
dim =1000
step = 1000
nodes = [1024, 128]
att = 1001
dct = False
netFile = '../../NNs/L/L_0'


def get_values(lines):
    x = []
    y = []
    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])

    return np.mean(np.array(y)), x

filename = '/home/weihao/tmp/L/L_11.csv' #sys.argv[1]

output = tf.placeholder(tf.float32, [None, 1])
input = tf.placeholder(tf.float32, [None, att])
learning_rate = tf.placeholder(tf.float32, shape=[])

avg_file = os.path.join('/home/weihao/Projects/p_files/L10000', 'Avg.p')
with open(avg_file, 'r') as fp:
    A = pickle.load(fp)
avgf = A[0]
stdf = A[1]
avg0 = A[2]

net = sNet3({'data': input})
net.real_setup(nodes, 1)
xy = net.layers['output']

with open(filename, 'r') as fp:
    line0 = fp.readlines()


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    saver.restore(sess, netFile)
    print len(line0)
    start = 0
    seg_step = 100000

    for start in range(0, len(line0), seg_step):

        lines = line0[start:start+SEG]
        avg, x = get_values(lines)

        a = 0
        r = []
        while a<len(x)-seg:

            features = l_utils.feature_final(x[a:a+seg], dct, dim)
            features = (features-avgf)/stdf
            features = features.reshape((1, len(features)))
            feed = {input: features}
            results = sess.run(xy, feed_dict=feed)[:, 0]
            a += step
            r.append(results[0])

        print avg, len(r), np.mean(r)+avg0, np.median(r)+avg0
