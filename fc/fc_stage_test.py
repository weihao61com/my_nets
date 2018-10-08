import sys
from fc_dataset import *
import random

sys.path.append( '..')
from utils import Utils

HOME = '/home/weihao/Projects/Posenet/'

if __name__ == '__main__':

    config_file = "config_stage.json"

    data_type = 'te'
    if len(sys.argv)>2:
        data_type = sys.argv[2]

    if len(sys.argv)>3:
        config_file = sys.argv[3]

    loop = 1
    if len(sys.argv)>1:
        loop = int(sys.argv[1])

    js = Utils.load_json_file(config_file)

    te_data = []
    for key in js:
        if key.startswith(data_type):
            te_data.append(HOME + js[key])

    netFile = HOME + js['netTest'] + '/fc'
    batch_size = int(js['batch_size'])
    feature_len = int(js['feature'])

    num_output = int(js["num_output"])

    te_set = DataSet(te_data, batch_size, feature_len)

    num_att = te_set.sz[1]
    print "input shape", te_set.sz, 'feature', feature_len

    input = tf.placeholder(tf.float32, [None, feature_len* num_att])
    stage_input = tf.placeholder(tf.float32, [None, num_att+num_output])

    output = tf.placeholder(tf.float32, [None, num_output])

    if num_output==3:
        net = sNet3({'data': input})
        stage_net = sNet3_stage({'data': stage_input})
    else:
        net = sNet1({'data': input})
        stage_net = None

    xy = net.layers['output']
    xy_stage = stage_net.layers['output_stage']

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, netFile)

        t00 = datetime.datetime.now()

        te_set.reshuffle_data()
        for bucket in te_set.data:
            for d in bucket:
                if random.random()>0.1:
                    continue
                I = d[0]
                while len(I) < feature_len:
                    I = np.concatenate((I, I))
                I = I[:feature_len].reshape((1,feature_len* num_att))
                T = d[1][:num_output]
                result = sess.run(xy, feed_dict={input: I})

                RT = []
                for p in d[0]:
                    J = np.concatenate((result[0], p))
                    J = J.reshape((1, len(J)))
                    RT.append(sess.run(xy_stage, feed_dict={stage_input: J})[0])

                print T, result[0], np.average(RT, axis=0), np.std(RT, axis=0)
