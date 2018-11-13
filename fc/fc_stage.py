import sys
from fc_dataset import *
import tensorflow as tf
import datetime

HOME = '{}/Projects/'.format(os.getenv('HOME'))
sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils
from fc_dataset import DataSet

if __name__ == '__main__':

    config_file = "config_stage.json"

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
    stage = js["stage"]
    num_output = int(js["num_output"])
    nodes0 = map(int, js["nodes0"].split(','))
    nodes1 = map(int, js["nodes1"].split(','))
    assert(nodes0[-2]==nodes1[-2])
    num_ref = nodes0[-2]
    nodes0.append(num_output)
    nodes1.append(num_output)

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '/fc'

    tr = DataSet(tr_data, batch_size, feature_len)
    te = DataSet(te_data, batch_size, feature_len)

    num_att = te.sz[1]
    iterations = 10000
    loop = 100
    print "Attribute", num_att, "LR", lr, 'feature', feature_len

    input0 = tf.placeholder(tf.float32, [None, feature_len * num_att])
    input1 = tf.placeholder(tf.float32, [None, num_att + num_ref])
    output = tf.placeholder(tf.float32, [None, num_output])

    net = sNet3_2({'data0': input0})
    net.real_setup(nodes0)
    stage_net = sNet3_stage({'data1': input1})
    stage_net.real_setup(nodes1)

    xy = net.layers['output0']
    xy_ref = net.layers['fc0_1']
    xy_stage = stage_net.layers['output1']
    st_ref = stage_net.layers['fc1_1']

    loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))
    loss_stage = tf.reduce_sum(tf.square(tf.subtract(xy_stage, output)))

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                        beta2=0.999, epsilon=0.00000001,
                        use_locking=False, name='Adam').\
        minimize(loss)
    opt_stage = tf.train.AdamOptimizer(learning_rate=lr/10, beta1=0.9,
                        beta2=0.999, epsilon=0.00000001,
                        use_locking=False, name='Adam').\
        minimize(loss_stage)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        if renetFile:
            saver.restore(sess, renetFile)

        t00 = datetime.datetime.now()

        str1 = ''
        for a in range(iterations):

            tr_pre_data = tr.prepare_stage()
            tr_loss, tr_median, tr_r2 = \
                run_stage_data(tr_pre_data, input0, input1, sess, xy, xy_ref,
                               xy_stage, st_ref)

            te_pre_data = te.prepare_stage()
            te_loss, te_median, te_r2= \
                run_stage_data(te_pre_data, input0, input1, sess, xy, xy_ref,
                               xy_stage, st_ref)

            t1 = datetime.datetime.now()
            str = "iteration: {0} {1:.3f} {2:f} {3:f} {4:f} {5:f} {6:f} {7:f} {8:f} {9:f}".\
                      format(a*loop/1000.0, (t1 - t00).total_seconds()/3600.0,
                             tr_loss, te_loss,
                             tr_median, te_median,
                             tr_r2[0], te_r2[0],
                             tr_r2[1], te_r2[1]
                             )
            print str, str1

            total_loss = 0
            nt = 0
            for _ in range(loop):
                tr_pre_data = tr.prepare_stage()
                while tr_pre_data is not None:
                    for b in tr_pre_data:
                        if stage == 0:
                            feed = {input0: b[0], output: b[2]}
                            _, l = sess.run([opt, loss], feed_dict=feed)
                            total_loss += l
                            nt += len(b[2])
                        else:
                            feed = {input0: b[0]}
                            result = sess.run(xy_ref, feed_dict=feed)
                            b1 = b[1]
                            n1 = b1.shape[1]

                            for a in range(n1):
                                input_array = np.concatenate((result, b1[:,a,:]), axis=1)
                                feed = {input1: input_array, output: b[2]}
                                if a < n1-1:
                                    _, result = sess.run([opt_stage, st_ref], feed_dict=feed)
                                else:
                                    _, l = sess.run([opt_stage, loss_stage], feed_dict=feed)
                                    total_loss += l
                                    nt += len(b[2])
                    tr_pre_data = tr.get_next()
            str1 = '{}'.format(total_loss/nt)

            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)

