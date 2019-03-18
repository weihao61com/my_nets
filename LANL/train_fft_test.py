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

SEGMENT = 150000
#
#
# SEG = 10000
# CV = 5
# dct = False
# dim = 200
# threads = 2
# location = '/home/weihao/tmp/L'
# out_location = '/home/weihao/Projects/p_files/L/L_{}'
# nodes = [1024, 128]
# eval_file = '/home/weihao/tmp/fit.csv'
# step = 1000
# att = dim+1


def get_values(lines):
    x = []
    y = []
    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])

    return np.mean(np.array(y)), x


def fft_test(config):
    cfg = Utils.load_json_file(config)
    eval_file = cfg['eval_file'].format(HOME)
    location = cfg['location'].format(HOME)
    SEG = cfg['SEG']
    dct = cfg['dct']>0
    dim = cfg['dim']
    step =cfg['testing_step']

    fp0 = open(eval_file, 'w')
    CV = cfg['CV']

    for c in range(CV):

        netFile = cfg['netFile'].format(HOME, c)
        files = glob.glob(os.path.join(location, 'L_*.csv'))
        idx = l_utils.rdm_ids(files)

        out_location = cfg['out_location'].format(HOME, c)
        par_location = os.path.dirname(out_location)
        avg_file = os.path.join(par_location, 'Avg.p')
        with open(avg_file, 'r') as fp:
            A = pickle.load(fp)
        avgf = A[0]
        stdf = A[1]
        # avg0 = A[2]
        att = len(avgf)

        input = tf.placeholder(tf.float32, [None, att])
        nodes = map(int, cfg['nodes'].split(','))

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
                    seg_step = 100000
                    t_scale = float(line0[0].split(',')[1])

                    for start in range(0, len(line0), seg_step):

                        lines = line0[start:start+SEGMENT]
                        avg, x = get_values(lines)
                        avg /= t_scale
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
                        if len(r)>0:
                            fp0.write('{},{},{},{},{},{}\n'.
                                      format(c, avg, len(r), np.mean(r),
                                             np.median(r), np.std(r)))
                        # print c, avg, len(r), np.mean(r)+avg0, np.median(r)+avg0
        tf.reset_default_graph()

    fp0.close()


if __name__ == '__main__':
    config = 'config.json'
    fft_test(config)
