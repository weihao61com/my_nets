import sys
import os
#from fc_dataset import *
import tensorflow as tf
import datetime
from sortedcontainers import SortedDict
from LANL_Utils import l_utils, sNet3, HOME

import glob
import numpy as np
import pickle
import random
from scipy import fftpack, fft
from multiprocessing.dummy import Pool as ThreadPool


HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))
from network import Network
from utils import Utils


def create_T_F(y, x, t0):
    t = np.average(y) / t0
    # if t > 0.1 and t < 7.0:
    f = x
    return t, f


def create_feature(f, length, L_step):
    f = np.array(f)
    df = f.reshape((L_step, length/L_step))
    sz = df.shape
    for a in range(sz[0]):
        df[a, :] = abs(fft(df[a, :]))
    df = df[:, 1:sz[1]/2+1]
    d = np.mean(df, 0)
    return d

def process(c):
    f = c[0]
    step = c[1]
    gen = c[2]
    cfg = c[3]
    SEG = cfg['SEG']
    L_step = cfg['L_step']
    length = SEG*L_step
    #length = c[3]
    #L_step = c[4]
    tmp = cfg['tmp']
    basename = os.path.basename(f)[:-4]
    basename = os.path.join(HOME, tmp, basename)
    x, y = l_utils.load_data(f)
    t0 = 1 # y[0]
    print f, len(x) / l_utils.SEGMENT, basename, t0
    N = len(x) / step

    for g in range(gen):
        T = []
        F = []
        rps = np.random.randint(0, len(x) - length - 1, N)
        for r in rps:
            t, f = create_T_F(y[r:r + length], x[r:r + length], t0)
            T.append(t)
            F.append(create_feature(f, length, L_step))
        fn = basename + '_{}.p'.format(g)
        print g, len(T)
        if len(T) > 0:
            with open(fn, 'w') as fp:
                pickle.dump((T, F), fp)


def gen_data(cfg):
    files = glob.glob(os.path.join(cfg['location'].format(HOME), 'L_*.csv'))
    print 'Total file', len(files)
    ids = l_utils.rdm_ids(files)

    file2 = []
    for f in ids:
        if ids[f] > -1:
            file2.append(f)

    gen = 5
    step = 10000

    cg = []
    for f in files:
        #for a in range(cfg['CV']):
        cg.append([f, step, gen, cfg])

    if cfg['threads'] > 1:
        print 'processing thread ', len(cg), cfg['threads']
        pool = ThreadPool(cfg['threads'])
        pool.map(process, cg)
        pool.close()
        pool.join()
    else:
        for c in cg:
            process(c)


def get_id(filename):
    ids = os.path.basename(filename).split('.')[0]
    ids = ids.split('_')
    return ids[0]+'_'+ids[1]


def generate_data(file1, prob, pth='tmp0/L*.p'):

    ids = []
    for f in file1:
        ids.append(get_id(f))

    files = glob.glob(os.path.join(HOME, pth))
    T = []
    F = []
    random.shuffle(files)
    for f in files:
        id = get_id(f)
        if id in ids:
            if random.random() < prob:
                # print f
                with open(f, 'r') as fp:
                    A = pickle.load(fp)
                    T = T + A[0]
                    F = F + A[1]
            if prob > 1.0:
                ids.remove(id)

    T = np.array(T)
    F = np.array(F)
    # print 'GD', len(T)
    #    , np.min(T), np.max(T), np.average(T), np.std(T),\
    #    np.average(F), np.std(F), np.min(F), np.max(F)
    return T, F

def get_avg_file(tr, avg_file):
    av = np.mean(tr, 0)
    st = np.std(tr, 0)
    with open(avg_file, 'w') as fp:
        pickle.dump((av,st), fp)

    return

def run_data(data, inputs, sess, xy, filename, cfg):
    truth = data[0]
    features = data[1]

    feed = {inputs: features}
    results = sess.run(xy, feed_dict=feed)[:, 0]

    if filename is not None:
        with open(filename, 'w') as fp:
            skip = int(len(truth)/2000)
            if skip==0:
                skip=1
            for a in range(len(truth)):
                if a%skip==0:
                    fp.write('{},{}\n'.format(truth[a], results[a]))

    return np.mean(np.abs(results-truth))

def avg_file_name(p):
    basename = os.path.basename(p)
    pathname = os.path.dirname(p)
    return pathname + '_' + basename+'_avg.p'

def avg_correction(data, avg_file):
    with open(avg_file, 'r') as fp:
        A = pickle.load(fp)
    av = A[0]
    st = A[1]
    out = []
    for b in range(data.shape[0]):
        d = data[b, :] - av
        d = d / st
        out.append(d)
    return np.array(out)

def get_testing_data(files, cfg):
    L_step = cfg['L_step']
    SEG = cfg['SEG']
    out = []
    for f in files:
        x, y = l_utils.load_data(f)
        t0 = 1 #y[0]
        print f, len(x) / l_utils.SEGMENT, t0
        for r in range(0, len(x), l_utils.SEGMENT):
            t, f = create_T_F(y[r:r + l_utils.SEGMENT], x[r:r + l_utils.SEGMENT], t0)
            if len(f) != l_utils.SEGMENT:
                break
            fs = []
            if L_step!=300:
                step = SEG #int(len(f)/200)
                for a in range(0, len(f), step):
                    f0 = f[a:a+SEG*L_step]
                    if len(f0)==SEG*L_step:
                        f0 = abs(fft(f0))[1:len(f0) / 2 + 1]
                        d = f0.reshape((SEG / 2, L_step))
                        d = np.mean(d, 1)
                        fs.append(d)
                        # fs.append(abs(fft(f0))[1:len(f0) / 2])
                    else:
                        break
            else:
                length = L_step * SEG
                fs.append(create_feature(f, length, L_step))
            out.append((t, np.array(fs)))
    return out


def run_testing(data, inputs, sess, xy, out_file, cfg):

    rs = []


    for d in data:
        features = d[1]
        feed = {inputs: features}
        results = sess.run(xy, feed_dict=feed)[:, 0]
        rs.append((d[0], len(results), np.median(results), np.mean(results), np.std(results)))

    rs = np.array(rs)
    print 'total data', len(rs)

    if out_file is not None:
        with open(out_file, 'w') as fp:
            for r in rs:
                fp.write('{},{},{},{},{}\n'.format(r[0], r[1], r[2], r[3], r[4]))

    rs = np.array(rs)
    print np.mean(np.abs(rs[:, 0] - rs[:, 2])), np.mean(np.abs(rs[:, 0] - rs[:, 3]))

def train(c, cfg, te=None):

    files = glob.glob(os.path.join(cfg['location'].format(HOME), 'L_*.csv'))
    ids = l_utils.rdm_ids(files)

    file1 = []
    file2 = []
    for f in ids:
        if ids[f] == c:
            file1.append(f)
        else:
            file2.append(f)

    nodes = map(int, cfg['nodes'].split(','))
    netFile = cfg['netFile']

    lr = cfg['lr']
    iterations = 10000
    loop = 1
    batch_size = 100
    cntn = cfg['cntn']>0

    print 'CV', c

    if te is None:
        data2 = generate_data(file2, .1, cfg['tmp'] + '/L*.p')
        data1 = generate_data(file1, 1.1, cfg['tmp'] + '/L*.p')
        att = data2[1].shape[1]
    else:
        data1 = get_testing_data(file1, cfg)
        att = data1[0][1].shape[1]


    output = tf.placeholder(tf.float32, [None, 1])
    input = tf.placeholder(tf.float32, [None, att])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    net = sNet3({'data': input})
    net.real_setup(nodes, 1)

    avg_file = avg_file_name(cfg['netFile']).format(HOME, c)
    if (not cntn) and (te is None):
        get_avg_file(data2[1], avg_file)

    if te is None:
        data1 = (data1[0], avg_correction(data1[1], avg_file))
        data2 = (data2[0], avg_correction(data2[1], avg_file))
    else:
        data2 = []
        for d in data1:
            data2.append((d[0], avg_correction(d[1], avg_file)))
        data1 = data2

    xy = net.layers['output']
    loss = tf.reduce_sum(tf.abs(tf.subtract(xy, output)))
    #loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9,
                    beta2=0.999, epsilon=0.00000001,
                    use_locking=False, name='Adam').\
        minimize(loss)
    # opt = tf.train.GradientDescentOptimizer(learning_rate=cfg.lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        if cntn or te is not None:
            saver.restore(sess, netFile.format(HOME, c))

        if te is not None:
            run_testing(data1, input, sess, xy, '{}/tmp/test.csv'.format(HOME), cfg)
            exit(0)

        t00 = datetime.datetime.now()
        st1 = ''
        for a in range(iterations):

            te_loss = run_data(data1, input, sess, xy, '{}/tmp/te.csv'.format(HOME), cfg)
            tr_loss = run_data(data2, input, sess, xy, '{}/tmp/tr.csv'.format(HOME), cfg)

            t1 = (datetime.datetime.now()-t00).seconds/3600.0
            str = "it: {0} {1:.3f} {2} {3} {4}".format(
                a*loop/1000.0, t1, lr, tr_loss, te_loss)
            print str, st1

            t_loss = 0
            t_count = 0
            data = generate_data(file2, .3, cfg['tmp'] + '/L*.p')

            truth = data[0]
            features = avg_correction(data[1], avg_file)

            length = len(truth)
            b0 = truth.reshape((length, 1))
            for lp in range(loop):
                for d in range(0, length, batch_size):
                    feed = {input: features[d:d+batch_size, :],
                            output: b0[d:d+batch_size, :],
                            learning_rate: lr
                    }
                    _, A = sess.run([opt, loss], feed_dict=feed)
                    t_loss += A
                    t_count += len(b0[d:d+batch_size])
            st1 = '{}'.format(t_loss/t_count)

            saver.save(sess, netFile.format(HOME, c))

    tf.reset_default_graph()


if __name__ == '__main__':

    config_file = "LH_config.json"
    if len(sys.argv)>1:
        config_file = sys.argv[1]

    test = None
    if len(sys.argv)>2:
        test = sys.argv[2]

    cfg = Utils.load_json_file(config_file)

    if test is None:
        train(0, cfg)
    elif test == 'gen':
        gen_data(cfg)
        train(0, cfg)
    elif test == '0':
        train(0, cfg, 't')

