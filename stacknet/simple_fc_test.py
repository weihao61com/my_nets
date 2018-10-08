import numpy as np
import random
import tensorflow as tf
import cv2
import datetime
import sys
#from o2_load import *
import random

#HOME = '/home/weihao/Projects/Posenet/'
from simple_fc_dataset import *
from utils import Utils

if __name__ == '__main__':
    config_file = "config.json"

    data_type = 'te'
    if len(sys.argv)>2:
        data_type = sys.argv[2]

    if len(sys.argv)>3:
        config_file = sys.argv[3]

    js = Utils.load_json_file(config_file)

    te_data = HOME + js[data_type]
    netFile = HOME + '/' + js['net'] + '/fc'
    batch_size = int(js['batch_size'])
    feature_len = int(js['feature'])
    num_output = int(js["num_output"])

    loop = 1
    if len(sys.argv)>1:
        loop = int(sys.argv[1])
    # data = sys.argv[1]

    te_set = DataSet(load_data(te_data))

    input = tf.placeholder(tf.float32, [None, feature_len*4])
    #output = tf.placeholder(tf.float32, [None, 2])

    if num_output==3:
        net = sNet({'data': input})
    else:
        net = sNet1({'data': input})

    xy = net.layers['output']

    #init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # with tf.Session(config=config) as sess:

    with tf.Session() as sess:
        #sess.run(init)
        saver.restore(sess, netFile)

        rst = {}
        truth = {}
        for _ in range(loop):
            nt = 0
            te_pre_data = te_set.prepare(False, nPar=feature_len, num_output=num_output)
            for b in te_pre_data:
                feed = {input: b[0]}
                result = sess.run(xy, feed_dict=feed)
                for a in range(len(result)):
                    if not nt in rst:
                        rst[nt] = []
                    rst[nt].append(result[a])
                    truth[nt] = b[1][a]
                    nt += 1

        fp = open('/home/weihao/tmp/test.csv', 'w')
        d = []
        for a in range(len(truth)):
            r, mm = cal_diff(truth[a], rst[a])
            if a==0:
                print truth[a]
                print mm, r
                for b in rst[a]:
                    print np.linalg.norm(b-truth[a]), b
            t = truth[a]
            #fp.write('{},{},{},{},{},{},{}\n'.
            #         format(t[0], t[1], t[2], mm[0], mm[1], mm[2], r))
            if random.random()<0.1:
                fp.write('{},{},{}\n'.
                     format(t[0], mm[0], r))
            d.append(r)
        fp.close()
        md = np.median(d)
        print len(truth), np.mean(d), md, np.sqrt(md)

