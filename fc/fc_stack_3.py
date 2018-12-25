import sys
from fc_dataset import *
import tensorflow as tf
import datetime

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils
from fc_dataset import DataSet

def run_data_stack_avg3(data, inputs, sess, xy, fname, att):
    rst_dic = {}
    truth_dic = {}
    length = 0
    for b in data:
        length = b[0].shape[1]/att
        feed = {inputs['input0']: b[0]}
        for a in range(length):
            feed[inputs['input{}'.format(a + 1)]] = b[0][:, att * a:att * (a + 1)]
        result = []
        for a in range(length+1):
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

    filename = '/home/weihao/tmp/{}.csv'.format(fname)
    if sys.platform == 'darwin':
        filename = '/Users/weihao/tmp/{}.csv'.format(fname)
    fp = open(filename, 'w')
    for id in rst_dic:
        dst = np.array(rst_dic[id])
        result = np.median(dst, axis=0)
        results.append(result)
        truth.append(truth_dic[id])
        t = truth_dic[id]
        if random.random() < 0.2:
            r = np.linalg.norm(t - result)
            mm = result[length - 1]
            if len(mm)==3:
                fp.write('{},{},{},{},{},{},{}\n'.
                     format(t[0], mm[0], t[1], mm[1], t[2], mm[2], r))
            else:
                fp.write('{},{},{}\n'.
                         format(t[0], mm[0], r))
    fp.close()

    return Utils.calculate_stack_loss_avg(np.array(results), np.array(truth))


if __name__ == '__main__':

    config_file = "config_stack_3.json"

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    cfg = Config(config_file)

    tr = DataSet(cfg.tr_data, cfg)
    te = DataSet(cfg.te_data, cfg)
    tr0 = DataSet([cfg.tr_data[0]], cfg)

    att = te.sz[1]
    iterations = 10000
    loop = cfg.loop
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
    net.real_setup(cfg.feature_len, cfg.nodes, num_out=cfg.num_output, att=att, verbose=False)

    xy = {}
    for a in range(cfg.feature_len+1):
        xy[a] = net.layers['output{}'.format(a)]

    ls = [tf.reduce_sum(tf.square(tf.subtract(xy[0], output)))]
    loss = None
    for x in range(1, cfg.feature_len+1):
        ll = tf.reduce_sum(tf.square(tf.subtract(xy[x], output)))
        if loss is None:
            loss = ll
        else:
            loss = loss + ll
        ls.append(ll)

    opt = tf.train.AdamOptimizer(learning_rate=cfg.lr, beta1=0.9,
                    beta2=0.999, epsilon=0.00000001,
                    use_locking=False, name='Adam').\
        minimize(loss)
    opt0 = tf.train.AdamOptimizer(learning_rate=cfg.lr*3, beta1=0.9,
                    beta2=0.999, epsilon=0.00000001,
                    use_locking=False, name='Adam').\
        minimize(ls[0])

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    t00 = datetime.datetime.now()

    with tf.Session() as sess:
        sess.run(init)
        if cfg.renetFile:
            saver.restore(sess, cfg.renetFile)

        str1 = ''
        for a in range(iterations):

            tr_pre_data = tr0.prepare()
            tr_loss, tr_median = run_data_stack_avg3(tr_pre_data, input_dic, sess, xy, 'tr', att)

            te_pre_data = te.prepare()
            te_loss, te_median = run_data_stack_avg3(te_pre_data, input_dic, sess, xy, 'te', att)

            t1 = datetime.datetime.now()
            str = "it: {0:.3f} {1:.3f}".format(a*loop/1000.0, (t1 - t00).total_seconds()/3600.0)
            s = 0
            while True:
                # for s in range(0, feature_len+1, 5  ):
                if s>cfg.feature_len:
                    s = cfg.feature_len
                str += " {0:.3f} {1:.3f} {2:.3f} {3:.3f} ".format(tr_loss[s], te_loss[s], tr_median[s], te_median[s])
                if s==cfg.feature_len:
                    break
                s += int(cfg.feature_len/2)
                if s>cfg.feature_len:
                    s = cfg.feature_len

            print str, str1

            tl3 = 0
            tl4 = 0
            tl5 = 0
            nt = 0
            for _ in range(loop):
                tr_pre_data = tr.prepare(multi=1)

                while tr_pre_data:
                    for b in tr_pre_data:
                        total_length = len(b[0])
                        for c in range(0, total_length, cfg.batch_size):
                            length = b[0].shape[1] - att * cfg.feature_len
                            feed = {input_dic['input0']: b[0][c:c + cfg.batch_size, :]}
                            for d in range(cfg.feature_len):
                                feed[input_dic['input{}'.format(d + 1)]] = \
                                    b[0][c:c + cfg.batch_size,  att * d: att * (d + 1)]
                            feed[output] = b[1][c:c + cfg.batch_size]
                            idx = int(cfg.feature_len/2)
                            # _ = sess.run([opt0], feed_dict=feed)
                            ll3,ll4,ll5, _, _ = sess.run([ls[0], ls[idx], ls[-1], opt0, opt],
                                                      feed_dict=feed)
                            tl3 += ll3
                            tl4 += ll4
                            tl5 += ll5
                            nt += len(b[0][c:c + cfg.batch_size])
                    tr_pre_data = tr.get_next()
            str1 = "{0:.3f} {1:.3f} {2:.3f}".format(tl3/nt, tl4/nt, tl5/nt)
            Utils.save_tf_data(saver, sess, cfg.netFile)


