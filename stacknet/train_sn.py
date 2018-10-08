# Import the converted model's class
import numpy as np
import tensorflow as tf
from stacknet import StackNet
from utils import Utils
import datetime
import sys
from o2_load import o2_data
import os
from utils_sn import Utils_SN


def main():
    config_file = "config.json"

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    js = Utils.load_json_file(config_file)
    location = js['directory']
    dataset = js['training_dataset']
    netFile = js['netFile']
    if not os.path.exists(netFile):
        os.mkdir(netFile)
    lr = 1e-8
    stack = int(js['stack'])


    retrain = None
    if 'retrain' in js:
        retrain = js['retrain']
        locs = os.path.dirname(retrain)
        refFile = '{}/ref.txt'.format(locs)
        init_ref = Utils_SN.read_ref(refFile)
        Nref = len(init_ref[0])
    else:
        Nref = 10
        init_ref = np.random.rand(1, Nref)

    refFile = '{}/ref.txt'.format(netFile)
    netFile = '{}/PNet'.format(netFile)

    with open(refFile, 'w') as fp:
        for a in init_ref[0]:
            fp.write("{} ".format(a))

    ref0 = tf.placeholder(tf.float32, [1, Nref])
    ins = {}
    for a in range(stack):
        ins[a] = tf.placeholder(tf.float32, [1, 2])
    outputs = tf.placeholder(tf.float32, [1, 4])

    in_dic = {}
    in_dic['ref0'] = ref0
    for a in range(stack):
        in_dic['input{}'.format(a)] = ins[a]
    net = StackNet(in_dic)

    net.set_stacK(stack)
    net.setup1()

    p_out = net.layers['ref{}'.format(stack)]

    loss = None
    for a in range(stack):
        p = net.layers['output{}'.format(a)]
        if loss is None:
            loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p, outputs)))) * 1.0
        else:
            loss = loss + tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p, outputs)))) * 1.0

    #loss = l1 + l2

    # opts = []
    #for A in range(3):
    ao = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999,
                                epsilon=0.00000001, use_locking=False, name='Adam')
    opts = ao.minimize(loss)
    #    lr /= 10

    # learning_rate = tf.placeholder(tf.float32, shape=[])
    # opt = tf.train.GradientDescentOptimizer(
    #    learning_rate=learning_rate).minimize(loss)
    # Set GPU options
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6833)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config_g = tf.ConfigProto(gpu_options=gpu_options)

    #device_count = {'CPU': 4},
    config = tf.ConfigProto(
        inter_op_parallelism_threads=7,
        intra_op_parallelism_threads=7
    )

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    rds = o2_data(location, dataset)


    iterations = 200000  # int(len(ds.images)*20.0/batch_size)
    print('interation={}'.format(iterations))

    with tf.Session(config=config) as sess:

        # Load the data
        sess.run(init)
        if retrain:
            saver.restore(sess, retrain)

        Model_variables = tf.GraphKeys.MODEL_VARIABLES
        Global_Variables = tf.GraphKeys.GLOBAL_VARIABLES
        av = tf.get_collection(Global_Variables)
        #for i in av:
        #    print (str(i) + '  -->  ' + str(i.eval()))

        #for v in tf.trainable_variables():
        #    print v.name

        v0 = None
        for v in tf.trainable_variables():
            print v.name
            v0 = v


        t0 = datetime.datetime.now()
        for a in range(iterations):
            total_loss = 0.0
            nt = 0
            event_gen = rds.event_gen(stack, True)
            for b in range(rds.len):
                evt = next(event_gen)
                ref = init_ref
                data_gen = evt.data_gen(True)
                losses = []

                for c in range(evt.len):
                    inputs, output = next(data_gen)
                    feed = {}
                    feed[ref0] = ref
                    feed[outputs] = output
                    for d in range(stack):
                        feed[ins[d]] = inputs[d]

                    l, num_outs = sess.run([loss, p_out], feed_dict=feed)
                    sess.run(opts, feed_dict=feed)
                    total_loss += l
                    nt += 1
                    ref = num_outs
                    losses.append(l)

                #if b%25==0 and a%400==0:
                #    t1 = datetime.datetime.now()
                #    print('It {0}, evt {1:04d}:  Loss {2}   {3}'.
                #          format(a, b, np.mean(losses), t1-t0))
                #    t0 = t1

            if a%1000==0:
                saver.save(sess, netFile)
                t1 = datetime.datetime.now()
                print("Total losses {}  {}  {}".
                      format(a, total_loss / nt, t1-t0))
                t0 = t1

        saver.save(sess, netFile)
        print("Intermediate file saved at: " + netFile)
        print("Total losses {}  {}".format(a, total_loss / nt))


        total_loss = 0.0
        nt = 0
        event_gen = rds.event_gen(stack, False)
        for b in range(rds.len):
            evt = next(event_gen)
            ref = init_ref
            data_gen = evt.data_gen()
            losses = []

            for c in range(evt.len):
                inputs, output = next(data_gen)
                feed = {}
                feed[ref0] = ref
                feed[outputs] = output
                for d in range(stack):
                    feed[ins[d]] = inputs[d]

                l, num_outs = sess.run([loss, p_out], feed_dict=feed)
                total_loss += l
                nt += 1
                ref = num_outs
                losses.append(l)
            # print("{}:  {} {}".format(b, np.mean(losses), inputs[0][0]))
        print("Total losses: {}".format(total_loss / nt))


if __name__ == '__main__':
    main()
