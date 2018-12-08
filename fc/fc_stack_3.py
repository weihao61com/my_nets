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

def run_data_stack_avg3(data, inputs, sess, xy, fname):
    rst_dic = {}
    truth_dic = {}
    length = 0
    for b in data:
        length = b[0].shape[1]/4
        feed = {inputs['input0']: b[0]}
        for a in range(length):
            feed[inputs['input{}'.format(a + 1)]] = b[0][:, 4 * a:4 * (a + 1)]
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

    js = Utils.load_json_file(config_file)

    tr_data = []
    te_data = []
    for key in js:
        if key.startswith('tr'):
            tr_data.append(HOME + js[key])
        if key.startswith('te'):
            te_data.append(HOME + js['te'])

    netFile = HOME + 'NNs/' + js['net'] + '/fc'

    batch_size = js['batch_size']
    feature_len = js['feature']
    lr = js['lr']
    #stack = js['stack']
    num_output = js["num_output"]
    step = js["step"]
    #stage = js["stage"]
    t_scale = js['t_scale']
    #net_type = js['net_type']

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '/fc'

    tr = DataSet(tr_data, batch_size, feature_len)
    te = DataSet(te_data, batch_size, feature_len)
    tr.set_t_scale(t_scale)
    te.set_t_scale(t_scale)
    tr.set_num_output(num_output)
    te.set_num_output(num_output)

    att = te.sz[1]
    iterations = 10000
    loop = js["loop"]
    print "input attribute", att, "LR", lr, 'feature', feature_len

    inputs = {}

    inputs[0] = tf.placeholder(tf.float32, [None, feature_len*att])
    output = tf.placeholder(tf.float32, [None, num_output])
    for a in range(feature_len):
        inputs[a+1] = tf.placeholder(tf.float32, [None, att])

    input_dic = {}
    for a in range(feature_len+1):
        input_dic['input{}'.format(a)] = inputs[a]

    net = StackNet(input_dic)
    net.real_setup(feature_len, num_out=num_output, verbose=False)

    xy = {}
    for a in range(feature_len+1):
        xy[a] = net.layers['output{}'.format(a)]

    ls = []
    loss = None
    for x in range(feature_len+1):
        ll = tf.reduce_sum(tf.square(tf.subtract(xy[x], output)))
        if loss is None:
            loss = ll
        else:
            loss = loss + ll
        ls.append(ll)

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                    beta2=0.999, epsilon=0.00000001,
                    use_locking=False, name='Adam').\
        minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    t00 = datetime.datetime.now()

    with tf.Session() as sess:
        sess.run(init)
        if renetFile:
            saver.restore(sess, renetFile)

        str1 = ''
        for a in range(iterations):

            tr_pre_data = tr.prepare()
            tr_loss, tr_median = run_data_stack_avg3(tr_pre_data, input_dic, sess, xy, 'tr')

            te_pre_data = te.prepare()
            te_loss, te_median = run_data_stack_avg3(te_pre_data, input_dic, sess, xy, 'te')

            t1 = datetime.datetime.now()
            str = "it: {0:.3f} {1:.3f}".format(a*loop/1000.0, (t1 - t00).total_seconds()/3600.0)
            s = 0
            while True:
                # for s in range(0, feature_len+1, 5  ):
                str += " {0:.3f} {1:.3f} {2:.3f} {3:.3f} ".format(tr_loss[s], te_loss[s], tr_median[s], te_median[s])
                if s==feature_len:
                    break
                s += int(feature_len/2)
                if s>feature_len:
                    s = feature_len

            print str, str1

            tl3 = 0
            tl4 = 0
            tl5 = 0
            nt = 0
            for _ in range(loop):
                tr_pre_data = tr.prepare(multi=100)

                while tr_pre_data:
                    for b in tr_pre_data:
                        total_length = len(b[0])
                        for c in range(0, total_length, step):
                            length = b[0].shape[1] - att * feature_len
                            feed = {input_dic['input0']: b[0][c:c + step, :]}
                            for d in range(feature_len):
                                feed[input_dic['input{}'.format(d + 1)]] = \
                                    b[0][c:c + step,  4 * d: 4 * (d + 1)]
                            feed[output] = b[1][c:c + step]
                            idx = int(feature_len/2)
                            ll3,ll4,ll5, _ = sess.run([ls[0], ls[idx], ls[-1], opt],
                                                      feed_dict=feed)
                            tl3 += ll3
                            tl4 += ll4
                            tl5 += ll5
                            nt += len(b[0][c:c + step])
                    tr_pre_data = tr.get_next()
            str1 = "{0:.3f} {1:.3f} {2:.3f}".format(tl3/nt, tl4/nt, tl5/nt)
            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)

