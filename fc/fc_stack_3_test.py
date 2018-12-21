import sys
from fc_dataset import *
import tensorflow as tf
import datetime
import fc_const

from utils import Utils

if __name__ == '__main__':

    data_type = 'te'
    if len(sys.argv)>2:
        data_type = sys.argv[2]

    loop = 1
    multi = -1
    if len(sys.argv)>1:
        loop = int(sys.argv[1])

    config_file = "config_stack_3.json"
    if len(sys.argv)>3:
        config_file = sys.argv[3]
    cfg = Config(config_file)

    data = cfg.te_data
    if data_type == 'tr':
        data = cfg.tr_data

    te = DataSet(data, cfg.batch_size, cfg.feature_len)
    te.set_t_scale(cfg.t_scale)
    te.set_num_output(cfg.num_output)

    att = te.sz[1]
    iterations = 10000
    print "input attribute", att, "LR", cfg.lr, 'feature', cfg.feature_len

    inputs = {}

    inputs[0] = tf.placeholder(tf.float32, [None, cfg.feature_len*att])
    output = tf.placeholder(tf.float32, [None, cfg.num_output])
    for a in range(cfg.feature_len):
        inputs[a+1] = tf.placeholder(tf.float32, [None, att])

    input_dic = {}
    for a in range(cfg.feature_len+1):
        input_dic['input{}'.format(a)] = inputs[a]

    net = StackNet(input_dic)
    # net.real_setup(cfg.feature_len, num_out=cfg.num_output, verbose=False)
    #net.real_setup(cfg.feature_len, cfg.nodes, num_out=cfg.num_output, verbose=False)
    net.real_setup(cfg.feature_len, cfg.nodes, num_out=cfg.num_output, att=att, verbose=False)

    xy = {}
    for a in range(cfg.feature_len+1):
        xy[a] = net.layers['output{}'.format(a)]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    t00 = datetime.datetime.now()

    with tf.Session() as sess:
        sess.run(init)
        print 'NET: ', cfg.netTest
        saver.restore(sess, cfg.netTest)

        rst_dic = {}
        truth_dic = {}
        for a in range(loop):
            data = te.prepare(multi=multi)

            for b in data:
                feed = {input_dic['input0']: b[0]}
                for a in range(cfg.feature_len):
                    feed[input_dic['input{}'.format(a + 1)]] = b[0][:, att * a:att * (a + 1)]
                result = []
                for a in range(0, cfg.feature_len+1, cfg.feature_len/2):
                    r = sess.run(xy[a], feed_dict=feed)
                    result.append(r)
                result = np.array(result)
                for a in range(len(b[2])):
                    if not b[2][a] in rst_dic:
                        rst_dic[b[2][a]] = []
                    rst_dic[b[2][a]].append(result[:, a, :])
                    truth_dic[b[2][a]] = b[1][a]

        results = []
        truth = []

        #fname = 'test'
        #filename = '/home/weihao/tmp/{}.csv'.format(fname)
        #if sys.platform == 'darwin':
        #    filename = '/Users/weihao/tmp/{}.csv'.format(fname)
        #fp = open(filename, 'w')
        for id in rst_dic:
            dst = np.array(rst_dic[id])
            result = np.median(dst, axis=0)
            # result = np.mean(dst, axis=0)
            results.append(result)
            truth.append(truth_dic[id])
            t = truth_dic[id]
            #if random.random() < 0.2:
            #    r = np.linalg.norm(t - result[cfg.feature_len])
            #    mm = result[cfg.feature_len]
            #    if len(mm)==3:
            #        fp.write('{},{},{},{},{},{},{}\n'.
            #             format(t[0], mm[0], t[1], mm[1], t[2], mm[2], r))
            #    else:
            #        fp.write('{},{},{}\n'.
            #                 format(t[0], mm[0], r))
        #fp.close()

        M, L = Utils.calculate_stack_loss_avg(np.array(results), np.array(truth))
        for a in range(len(M)):
            print a, M[a], L[a]


