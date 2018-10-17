import sys
from fc_dataset import *
import tensorflow as tf
import datetime

sys.path.append( '..')
from utils import Utils

HOME = '/home/weihao/Projects/'

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

    netFile = HOME + js['net'] + '/fc'
    batch_size = int(js['batch_size'])
    feature_len = int(js['feature'])
    lr = float(js['lr'])

    num_output = int(js["num_output"])

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + '/' + js['retrain'] + '/fc'

    stage_dup = 1
    N_layer2 = 64

    tr = DataSet(tr_data, batch_size, feature_len+stage_dup)
    te = DataSet(te_data, batch_size, feature_len+stage_dup)

    num_att = te.sz[1]
    iterations = 10000
    loop = 10
    print "input shape", te.sz, "LR", lr, 'feature', feature_len

    input       = tf.placeholder(tf.float32, [None, feature_len* num_att])
    stage_input = tf.placeholder(tf.float32, [None, num_att*stage_dup+N_layer2])
    output      = tf.placeholder(tf.float32, [None, num_output])

    net = sNet3_2({'data': input})
    stage_net = sNet3_stage({'data': stage_input})

    xy = net.layers['output']
    xy_fc2 = net.layers['fc2']
    xy_stage = stage_net.layers['output_stage']
    st_fc2 = stage_net.layers['fc2_stage']

    #loss = tf.reduce_sum(tf.square(tf.square(tf.subtract(xy, output))))
    loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))
    loss_stage = tf.reduce_sum(tf.square(tf.subtract(xy_stage, output)))

    opt = tf.train.AdamOptimizer(learning_rate=lr/40, beta1=0.9,
                        beta2=0.999, epsilon=0.00000001,
                        use_locking=False, name='Adam').\
        minimize(loss)
    opt_stage = tf.train.AdamOptimizer(learning_rate=lr/100, beta1=0.9,
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

        for a in range(iterations):

            tr_pre_data = tr.prepare()
            total_loss, tr_median, tr_r2 = \
                run_stage_data(tr_pre_data, input, sess, xy, stage_input,
                               xy_stage, stage_dup, xy_fc2, st_fc2)

            te_pre_data = te.prepare()
            te_loss, te_median, te_r2= \
                run_stage_data(te_pre_data, input, sess, xy, stage_input,
                               xy_stage, stage_dup, xy_fc2, st_fc2)

            t1 = datetime.datetime.now()
            str = "iteration: {0:d} {1:f} {2:f} {3:f} {4:f} {5:f} {6:f} {7:f} {8:f} {9:f}".\
                      format(a*loop, total_loss, te_loss, te_loss-total_loss,
                             tr_median, te_median, tr_r2[0][0],tr_r2[0][1],
                             te_r2[0][0], te_r2[0][1]
                             ) + ' time {}'.format(t1 - t00)
            print str
            t00 = t1

            for _ in range(loop):
                tr_pre_data = tr.prepare() #.get()
                while tr_pre_data is not None:
                    for b in tr_pre_data:
                        b_array = b[0][:, :-stage_dup*num_att]
                        sess.run(opt, feed_dict={input: b_array, output: b[1]})
                        result = sess.run(xy, feed_dict={input: b_array})
                        length = b[0].shape[1]
                        d_truth = b[1] - result
                        fc_input = sess.run(xy_fc2, feed_dict={input: b_array})
                        start = length - stage_dup * num_att
                        for evt in range(stage_dup):
                            input_array = np.concatenate(
                                (fc_input, b[0][:, start + evt * num_att: start + (evt + 1) * num_att]), axis=1)
                            sess.run(opt_stage, feed_dict={stage_input: input_array, output: d_truth})
                            fc_input = sess.run(st_fc2, feed_dict={stage_input: input_array})

                    tr_pre_data = tr.get_next()

            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)

