# Import the converted model's class
import sys
sys.path.append('/home/weihao/posenet/paranet')
import numpy as np
#import os
import tensorflow as tf
#from tensorflow.python.ops import rnn, rnn_cell
#from posenet import GoogLeNet as PoseNet
#import glob
from utils import Utils
from utils_sn import Utils_SN
from o2_load import o2_data
from stacknet import StackNet
from sortedcontainers import SortedDict

def main():
    import sys
    config_file = "config.json"

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    js = Utils.load_json_file(config_file)
    location = js['directory']
    dataset = js['testing_dataset']
    netFile = js['netTest']

    refFile = '{}/ref.txt'.format(netFile)
    netFile = '{}/PNet'.format(netFile)

    init_ref = Utils_SN.read_ref(refFile)
    Nref = init_ref.shape[1]
    ref0 = tf.placeholder(tf.float32, [1, Nref])
    outputs = tf.placeholder(tf.float32, [1, 4])

    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': -1},
        allow_soft_placement=True,
        log_device_placement=False,
        inter_op_parallelism_threads=3,
        intra_op_parallelism_threads=3
    )

    rds = o2_data(location, dataset)

    stack = int(js['stack'])
    ins = {}
    for a in range(stack):
        ins[a] = tf.placeholder(tf.float32, [1, 2])

    in_dic = {}
    in_dic['ref0'] = ref0
    for a in range(stack):
        in_dic['input{}'.format(a)] = ins[a]
    net = StackNet(in_dic)

    net.set_stacK(stack)
    net.setup()

    outputs = tf.placeholder(tf.float32, [1, 4])

    loss = None
    for a in range(stack):
        p = net.layers['output{}'.format(a)]
        if loss is None:
            loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p, outputs)))) * 1.0
        else:
            loss = loss + tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p, outputs)))) * 1.0

    p_out = net.layers['ref{}'.format(stack)]

    init = tf.global_variables_initializer()  # initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session(config=session_conf) as sess:

        # Load the data
        sess.run(init)
        saver.restore(sess, netFile)
        #
        # vs = tf.trainable_variables()
        # print "total val", len(vs)
        # for v in vs:
        #     #if v.name.startswith('bia'):
        #     vals = sess.run(v)
        #     if v.name.startswith('bia'):
        #         print v.name, vals.shape, vals[:5]
        #     else:
        #         print v.name, vals.shape, vals[0, :5]
        # #var = [v for v in tf.trainable_variables() if v.name == "tower_2/filter:0"][0]

        event_gen = rds.event_gen(stack, False)

        ms = SortedDict()
        for b in range(rds.len):
            evt = next(event_gen)
            ref = init_ref
            data_gen = evt.data_gen()
            losses = []
            for c in range(evt.len):
                if c not in ms:
                    ms[c] = []
                inputs, output = next(data_gen)
                feed = {}
                feed[ref0] = ref
                feed[outputs] = output
                for d in range(stack):
                    feed[ins[d]] = inputs[d]

                l, num_outs = sess.run([loss, p_out], feed_dict=feed)
                ms[c].append(l)

                ref = num_outs
                losses.append(l)
            # print("{}:  {} {}".format(b, np.mean(losses), inputs[0][0]))
        for c in ms:
            print b, c, np.mean(ms[c]), len(ms[c])



if __name__ == '__main__':
    import sys

    main()
