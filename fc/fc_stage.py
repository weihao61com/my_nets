import sys
from fc_dataset import *

sys.path.append( '..')
from utils import Utils

HOME = '/home/weihao/Projects/Posenet/'

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

    tr = DataSet(tr_data, batch_size, feature_len)
    te_set = DataSet(te_data, batch_size, feature_len)

    stage_dup = 5
    num_att = te_set.sz[1]
    iterations = 10000
    loop = 1
    print "input shape", te_set.sz, "LR", lr, 'feature', feature_len

    input = tf.placeholder(tf.float32, [None, feature_len* num_att])
    stage_input = tf.placeholder(tf.float32, [None, num_att*stage_dup+num_output])

    output = tf.placeholder(tf.float32, [None, num_output])


    if num_output==3:
        net = sNet3({'data': input})
        stage_net = sNet3_stage({'data': stage_input})
    else:
        net = sNet1({'data': input})
        stage_net =None


    xy = net.layers['output']
    xy_stage = stage_net.layers['output_stage']

    #loss = tf.reduce_sum(tf.square(tf.square(tf.subtract(xy, output))))
    loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))
    loss_stage = tf.reduce_sum(tf.square(tf.subtract(xy_stage, output)))

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
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
            total_loss, tr_median, tr_r2 = run_stage_data(tr_pre_data, input, sess,
                                                   xy, stage_input, xy_stage)

            te_pre_data = te_set.prepare()
            te_loss, te_median, te_r2= run_stage_data(te_pre_data, input, sess,
                                                xy, stage_input, xy_stage)

            t1 = datetime.datetime.now()
            str = "iteration: {0:d} {1:f} {2:f} {3:f} {4:f} {5:f} {6:f} {7:f}".format(
                a*loop, total_loss, te_loss, te_loss-total_loss,
                tr_median, te_median, tr_r2, te_r2) + ' time {}'.format(t1 - t00)
            print str
            t00 = t1

            for _ in range(loop):
                tr_pre_data = tr.prepare() #.get()
                while tr_pre_data is not None:
                    for b in tr_pre_data:
                        sess.run(opt, feed_dict={input: b[0], output: b[1]})
                        result = sess.run(xy, feed_dict={input: b[0]})
                        length = b[0].shape[1]
                        d_truth = b[1] - result
                        for evt in range(0, length, num_att*stage_dup):
                            input_array = np.concatenate((result, b[0][:, evt:evt+num_att*stage_dup]), axis=1)
                            sess.run(opt_stage, feed_dict={stage_input: input_array, output: d_truth})

                    tr_pre_data = tr.get_next()

            saver.save(sess, netFile)

        print netFile
        saver.save(sess, netFile)

