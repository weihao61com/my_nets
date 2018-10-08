# Import the converted model's class
#import numpy as np
#import random
#import tensorflow as tf
#import cv2
#import datetime
import sys
#from o2_load import *
#import Queue
#import copy

HOME = '/home/weihao/Projects/Posenet/'
#sys.path.append(HOME + 'my_nets/paranet')
from simple_fc_dataset import *
from utils import Utils

if __name__ == '__main__':

    config_file = "config.json"

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    js = Utils.load_json_file(config_file)

    tr_data = []
    for key in js:
        if key.startswith('tr'):
            tr_data.append(HOME + js[key])
    te_data = HOME + js['te']
    netFile = HOME + js['net'] + '/fc'
    batch_size = int(js['batch_size'])
    feature_len = int(js['feature'])
    lr = float(js['lr'])

    num_output = int(js["num_output"])

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + '/' + js['retrain'] + '/fc'

    tr = DataSet(load_data(tr_data), batch_size)
    te_set = DataSet(load_data(te_data), batch_size)

    sz_in = te_set.sz
    iterations = 10000
    loop = 10
    print "input shape", sz_in, "LR", lr, 'feature', feature_len

    input = tf.placeholder(tf.float32, [None, feature_len* sz_in[1]])
    output = tf.placeholder(tf.float32, [None, num_output])

    if num_output==3:
        net = sNet({'data': input})
    else:
        net = sNet1({'data': input})

    xy = net.layers['output']
    #loss = tf.reduce_sum(tf.square(tf.square(tf.subtract(xy, output))))
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

        for a in range(iterations):

            total_loss = 0
            #if queue.empty():
            #    queue = get_queue(tr, pool)
            tr_pre_data = tr.prepare(nPar=feature_len, num_output=num_output)
            for b in tr_pre_data:
                feed = {input: b[0], output: b[1]}
                ll = sess.run(loss, feed_dict=feed)
                total_loss += ll
            total_loss /= tr.length
            te_loss = 0

            te_pre_data = te_set.prepare(nPar=feature_len, num_output=num_output)
            for b in te_pre_data:
                feed = {input: b[0], output: b[1]}
                ll = sess.run(loss, feed_dict=feed)
                te_loss += ll
            te_loss /= te_set.length
            t1 = datetime.datetime.now()
            str = "iteration: {} {} {} {} time {}". \
                  format(a*loop, total_loss, te_loss, te_loss-total_loss, t1 - t00)
            print str
            t00 = t1

            for _ in range(loop):
                tr_pre_data = tr.prepare(nPar=feature_len, num_output=num_output) #.get()

                for b in tr_pre_data:
                    feed = {input: b[0], output: b[1]}
                    sess.run(opt, feed_dict=feed)

            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)

