# Import the converted model's class
import tensorflow as tf
from posenet import GoogLeNet as PoseNet
from utils import Utils
import sys

def main():
    config_file = "/home/weihao/posenet/paranet/config.json"

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    js = Utils.load_json_file(config_file)
    location = js['directory']
    batch_size = int(js['batch_size'])
    dataset = js['training_dataset']
    netFile = js['netFile']
    retrain = None
    if 'retrain' in js:
        retrain = js['retrain']
    rep = int(js['rep'])

    if rep>-1:
        netFile = '{}/output_{}/PNet'.format(netFile, rep)
    else:
        netFile = '{}/output_/PNet'.format(netFile)

    lr = 1e-3
    rg = 3

    images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    poses_x = tf.placeholder(tf.float32, [batch_size, 3])
    poses_q = tf.placeholder(tf.float32, [batch_size, 4])
    shift = tf.placeholder(tf.float32, [batch_size, 1, 1, 4])

    net = PoseNet({'data': images, 'shift': shift})

    #p1_x = net.layers['cls1_fc_pose_xyz']
    #p1_q = net.layers['cls1_fc_pose_wpqr']
    #p2_x = net.layers['cls2_fc_pose_xyz']
    #p2_q = net.layers['cls2_fc_pose_wpqr']
    p3_x = net.layers['cls3_fc_pose_xyz']
    p3_q = net.layers['cls3_fc_pose_wpqr']

    #l1_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_x, poses_x)))) * 0.3
    #l1_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_q, poses_q)))) * 150
    #l2_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_x, poses_x)))) * 0.3
    #l2_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_q, poses_q)))) * 150
    l3_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_x, poses_x)))) * 1
    l3_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_q, poses_q)))) * 100

    loss = l3_x + l3_q #l1_x + l1_q + l2_x + l2_q + l3_x + l3_q

    opts = []
    for A in range(rg):
        ao = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999,
                                    epsilon=0.00000001, use_locking=False, name='Adam')
        opts.append(ao.minimize(loss))
        lr /= 10

    # learning_rate = tf.placeholder(tf.float32, shape=[])
    # opt = tf.train.GradientDescentOptimizer(
    #    learning_rate=learning_rate).minimize(loss)
    # Set GPU options
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6833)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config_g = tf.ConfigProto(gpu_options=gpu_options)

    #device_count = {'CPU': 4},
    config = tf.ConfigProto(
        inter_op_parallelism_threads=6,
        intra_op_parallelism_threads=6
    )

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    rds = Utils.get_raw_data(location, dataset)

    iterations = 1000  # int(len(ds.images)*20.0/batch_size)
    print('interation={}'.format(iterations))

    with tf.Session(config=config) as sess:

        # Load the data
        sess.run(init)
        if retrain:
            saver.restore(sess, retrain)

        ds = Utils.get_data(rds, rep)
        data_gen = Utils.gen_data_batch(ds, batch_size)
        reload = 20 #len(ds.images)/batch_size

        print("Total images {}, rep {}, iter {}, rg {}".format(len(ds.images), rep, iterations, rg))

        for A in range(rg):
            import datetime
            t0 = datetime.datetime.now()
            for i in range(iterations):
                np_images, np_poses_x, np_poses_q, np_shift = next(data_gen)
                feed = {images: np_images, poses_x: np_poses_x,
                        poses_q: np_poses_q, shift: np_shift}  # , learning_rate: lr}

                sess.run(opts[A], feed_dict=feed)
                np_loss = sess.run(loss, feed_dict=feed)
                if (i + 1) % reload == 0:
                    t1 = datetime.datetime.now()
                    print("iteration: {} loss {} time {}".format(i, np_loss, t1 - t0))
                    t0 = t1
                    saver.save(sess, netFile)
                    #ds = Utils.get_data(rds, rep, True)
                    #data_gen = Utils.gen_data_batch(ds, batch_size)

            saver.save(sess, netFile)
            print("Intermediate file saved at: " + netFile)


if __name__ == '__main__':
    main()
