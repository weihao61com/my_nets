import sys
from fc_dataset import *
import tensorflow as tf
import datetime

HOME = '{}/Projects/'.format(os.getenv('HOME'))
sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils
from fc_dataset import DataSet
from network import Network
from fc_pyramid import PyraNet

if __name__ == '__main__':

    config_file = "config_pyra.json"

    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    js = Utils.load_json_file(config_file)

    te_data = []
    for key in js:
        if key.startswith('te'):
            te_data.append(HOME + js['te'])

    netFile = HOME + 'NNs/' + js['net'] + '/fc'

    batch_size = js['batch_size']
    feature_len = js['feature']
    base = js['base']
    step = js["step"]
    loop = 1
    num_output = 1
    t_scale = js['t_scale']

    renetFile = HOME + 'NNs/' + js['netTest'] + '/fc'

    te = DataSet(te_data, batch_size, feature_len * base)
    te.set_t_scale(t_scale)
    te.set_num_output(num_output)

    att = te.sz[1]
    print "input attribute", att, 'feature', feature_len

    inputs = {}
    input_dic = {}

    output = tf.placeholder(tf.float32, [None, num_output])
    for a in range(base):
        inputs[a] = tf.placeholder(tf.float32, [None, att*feature_len])
        input_dic['input{}'.format(a)] = inputs[a]

    net = PyraNet(input_dic)
    net.real_setup(feature_len)

    xy = {}
    for a in range(base + 1):
        xy[a] = net.layers['output{}'.format(a)]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    t00 = datetime.datetime.now()
    multi = -1

    with tf.Session() as sess:
        saver.restore(sess, renetFile)


        rst_dic = {}
        truth_dic = {}
        for _ in range(loop):
            te_pre_data = te.prepare(rd=False,multi=multi)
            for b in te_pre_data:

                length = b[0].shape[1] / base
                feed = {}  # inputs['input0']: b[0][:, :length]}
                for a in range(base):
                    feed[input_dic['input{}'.format(a)]] = b[0][:, length * a: length * (a + 1)]
                result = []
                for a in range(base + 1):
                    r = sess.run(xy[a], feed_dict=feed)
                    result.append(r)
                result = np.array(result)
                for a in range(len(b[2])):
                    if not b[2][a] in rst_dic:
                        rst_dic[b[2][a]] = []
                    rst_dic[b[2][a]].append(result[:, a, :])
                    truth_dic[b[2][a]] = b[1][a]

        rst = []
        truth = []

        filename = '/home/weihao/tmp/test.csv'
        if sys.platform == 'darwin':
            filename = '/Users/weihao/tmp/test.csv'
        fp = open(filename, 'w')
        for id in rst_dic:
            dst = np.array(rst_dic[id])
            result = np.median(dst, axis=0)
            rst.append(result)
            truth.append(truth_dic[id])
            t = truth_dic[id]
            if random.random() < 0.2:
                r = np.linalg.norm(t - result)
                mm = result[base - 1]
                if len(t) == 3:
                    fp.write('{},{},{},{},{},{},{}\n'.
                             format(t[0], mm[0], t[1], mm[1], t[2], mm[2], r))
                else:
                    fp.write('{},{}\n'.
                             format(t[0], mm[0]))

        fp = open('{}/../tmp/test.csv'.format(HOME), 'w')
        d = []
        for a in range(len(truth)):
            r, mm = cal_diff(truth[a], rst[a])
            if a==0:
                print truth[a], mm, r
                diff = []
                for b in rst[a]:
                    # print np.linalg.norm(b-truth[a]), b, truth[a]
                    diff.append(b-truth[a])
                diff = np.array(diff)
                for b in range(diff.shape[1]):
                    hist, bins = np.histogram(diff[:,b])


            t = truth[a]
            #fp.write('{},{},{},{},{},{},{}\n'.
            #         format(t[0], t[1], t[2], mm[0], mm[1], mm[2], r))
            if random.random()<0.2:
                if num_output==3:
                    fp.write('{},{},{},{},{},{},{}\n'.
                         format(t[0], mm[0],t[1], mm[1],t[2], mm[2], r))
                else:
                    fp.write('{},{},{}\n'.
                         format(t[0], mm[0], r))
            d.append(r*r)
        fp.close()
        md = np.median(d)
        print len(truth), np.mean(d), np.sqrt(md)
