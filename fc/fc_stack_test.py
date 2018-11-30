import sys
from fc_dataset import *
import tensorflow as tf
import datetime

sys.path.append('..')
from utils import Utils

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

if __name__ == '__main__':

    config_file = HOME + "/my_nets/fc/config_stack.json"

    data_type = 'te'
    if len(sys.argv) > 1:
        data_type = sys.argv[1]

    if len(sys.argv) > 2:
        config_file = sys.argv[2]

    multi = -1
    if len(sys.argv)>3:
        multi = int(sys.argv[3])

    js = Utils.load_json_file(config_file)

    te_data = []
    for key in js:
        if key.startswith(data_type):
            te_data.append(HOME + js[key])

    netFile = HOME + 'NNs/' + js['netTest'] + '/fc'

    batch_size = js['batch_size']
    feature_len = js['feature']
    stack = js['stack']
    num_output = js["num_output"]
    t_scale = js['t_scale']
    net_type = js['net_type']

    te = DataSet(te_data, batch_size, feature_len + stack)
    te.set_net_type(net_type)
    te.set_t_scale(t_scale)
    te.set_num_output(num_output)
    sz_in = te.sz

    print "input shape", sz_in, 'feature', feature_len

    inputs = {}

    inputs[0] = tf.placeholder(tf.float32, [None, feature_len * sz_in[1]])
    output = tf.placeholder(tf.float32, [None, num_output])
    for a in range(stack):
        inputs[a + 1] = tf.placeholder(tf.float32, [None, sz_in[1]])

    input_dic = {}
    for a in range(stack + 1):
        input_dic['input{}'.format(a)] = inputs[a]

    net = StackNet(input_dic)
    net.real_setup(stack, verbose=False, num_out=num_output)

    xy = {}
    for a in range(stack+1):
        xy[a] = net.layers['output{}'.format(a)]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, netFile)

        t00 = datetime.datetime.now()

        te_pre_data = te.prepare(multi=multi, rd=False)
        te_loss, te_median = run_data_stack_avg(te_pre_data, input_dic, sess, xy, stack, 'test')

        t1 = datetime.datetime.now()

        str = "it: {0:.2f}  ".format((t1 - t00).total_seconds() / 3600.0)
        for s in range(stack + 1):
            str += "  {0:.4f} {1:.4f}  ".format(te_loss[s], te_median[s])


        print str
