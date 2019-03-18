import matplotlib.pyplot as plt
import sys
import glob
import numpy as np
import pickle
import os
from LANL_Utils import l_utils, sNet3, HOME
import tensorflow as tf

sys.path.append('{}/my_nets'.format(HOME))
from utils import Utils


def run_refit(x, co):
    p = np.poly1d(co)
    return p(x)


def net_run(cfg, x):
    # eval_file = cfg['eval_file'].format(HOME)
    location = cfg['location'].format(HOME)
    SEG = cfg['SEG']
    dct = cfg['dct'] > 0
    dim = cfg['dim']
    step = cfg['testing_step']

    # fp0 = open(eval_file, 'w')
    CV = cfg['CV']

    out_location = cfg['out_location'].format(HOME, 0)
    par_location = os.path.dirname(out_location)
    avg_file = os.path.join(par_location, 'Avg.p')
    with open(avg_file, 'r') as fp:
        A = pickle.load(fp)
    avgf = A[0]
    stdf = A[1]
    avg0 = A[2]
    att = len(avgf)
    rst_file = cfg['refit_file'].format(HOME)
    with open(rst_file, 'r') as fp:
        refit = pickle.load(fp)

    output1 = []
    output2 = []
    for c in range(CV):

        netFile = cfg['netFile'].format(HOME, c)
        # files = glob.glob(os.path.join(location, 'L_*.csv'))
        # idx = l_utils.rdm_ids(files)

        input = tf.placeholder(tf.float32, [None, att])
        nodes = map(int, cfg['nodes'].split(','))

        net = sNet3({'data': input})
        net.real_setup(nodes, 1, False)
        xy = net.layers['output']

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, netFile)

            a = 0
            r = []
            while a <= len(x) - SEG:
                features = l_utils.feature_final(x[a:a + SEG], dct, dim)
                features = (features - avgf) / stdf
                features = features.reshape((1, len(features)))
                feed = {input: features}
                results = sess.run(xy, feed_dict=feed)[:, 0]
                a += step
                r.append(results[0])
            if len(r) > 0:
                v_mean = run_refit(np.mean(r) + avg0, refit[c][0])
                v_median = run_refit(np.median(r) + avg0, refit[c][0])
                #print(
                #    '{},{},{},{},{}'.format(c, len(r), v_mean, v_median,
                #                              np.std(r)))
                output1.append(v_mean)
                output2.append(v_median)
        tf.reset_default_graph()
    output1 = np.mean(np.array(output1))
    output2 = np.mean(np.array(output2))
    return output1, output2


def main5(cfg, filename, scale=10):
    x, y = l_utils.load_data(filename)

    if y is not None:
        length = len(x)
        step = length/20
        for a in range(0, length, step):
            X = np.array(x[a: a+l_utils.SEGMENT])
            Y = np.array(y[a: a+l_utils.SEGMENT])
            if len(X)==l_utils.SEGMENT:
                avg = np.mean(Y)
                A, B = net_run(cfg, X)
                print avg, A*scale, B*scale
    else:
        A, B = net_run(cfg, x)
        return A*scale, B*scale


def fft_run(config, filename):

    cfg = Utils.load_json_file(config)

    if filename.endswith('csv'):
        main5(cfg, filename)
    else:
        files = glob.glob(os.path.join(filename, '*.csv'))
        fp = open('{}/../LANL/rst.csv', 'r')
        for file in files:
            A, B = main5(cfg, file)
            fp.write('{},{},{}\n'.format(os.path.basename(file)[:-4], A, B))
            #print os.path.basename(file)[:-4], A, B
        fp.close()


if __name__ == '__main__':
    config = 'config.json'
    filename = '/home/weihao/tmp/L/L_11.csv'
    # filename = '/home/weihao/Downloads/test/seg_0a42ba.csv'
    if len(sys.argv)>1:
        filename = sys.argv[1]

    fft_run(config, filename)