# Import the converted model's class
import tensorflow as tf
import glob
import sys


def process(config_file, rep):
    from posenet import GoogLeNet as PoseNet
    from utils import Utils

    js = Utils.load_json_file(config_file)
    location = js['directory']
    batch_size = int(js['batch_size'])
    dataset = js['training_dataset']
    netFile_base = js['netFile']
    retrain = None
    if 'retrain' in js:
        retrain = js['retrain']

    if rep is None:
        rep = int(js['rep'])
    lr = 1e-3
    rg = 3
    classes = glob.glob(os.path.join(location, '*'))
    num_class = len(classes)

    images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    poses_x = tf.placeholder(tf.float32, [batch_size, 3])
    poses_q = tf.placeholder(tf.float32, [batch_size, 4])
    shift = tf.placeholder(tf.float32, [batch_size, 1, 1, num_class])

    net = PoseNet({'data': images, 'shift': shift})

    # p1_x = net.layers['cls1_fc_pose_xyz']
    # p1_q = net.layers['cls1_fc_pose_wpqr']
    # p2_x = net.layers['cls2_fc_pose_xyz']
    # p2_q = net.layers['cls2_fc_pose_wpqr']
    p3_x = net.layers['cls3_fc_pose_xyz']
    p3_q = net.layers['cls3_fc_pose_wpqr']

    # l1_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_x, poses_x)))) * 0.3
    # l1_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_q, poses_q)))) * 150
    # l2_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_x, poses_x)))) * 0.3
    # l2_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_q, poses_q)))) * 150
    l3_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_x, poses_x)))) * 1
    l3_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_q, poses_q)))) * 500.0

    loss = l3_x # + l3_q  # l1_x + l1_q + l2_x + l2_q + l3_x + l3_q

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
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6833)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config_g = tf.ConfigProto(gpu_options=gpu_options)

    # device_count = {'CPU': 4},
    config = tf.ConfigProto(
        inter_op_parallelism_threads=4,
        intra_op_parallelism_threads=4
    )

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    rds = Utils.get_raw_data_indoor(location, dataset, rep)

    with tf.Session(config=config) as sess:

        # Load the data
        sess.run(init)
        if retrain:
            saver.restore(sess, retrain)

        ds = Utils.get_data(rds, rep, r0=1,ss=480)
        data_gen = Utils.gen_data_batch(ds, batch_size)
        reload = len(ds.images)/batch_size

        epoch = 5000.0
        if rep == -1:
            epoch = 1000.0

        iterations = int(len(ds.images) * epoch / batch_size)

        print("Total images {}, rep {}, iter {}, reload {}".
              format(len(ds.images), rep, iterations, reload))

        for A in range(rg):
            import datetime
            if rep > -1:
                netFile = '{}/Net{}_{}/PNet'.format(netFile_base, A, rep)
            else:
                netFile = '{}/Net{}/PNet'.format(netFile_base, A)

            t0 = datetime.datetime.now()
            for i in range(iterations):
                np_images, np_poses_x, np_poses_q, np_shift = next(data_gen)
                feed = {images: np_images, poses_x: np_poses_x,
                        poses_q: np_poses_q, shift: np_shift}  # , learning_rate: lr}

                sess.run(opts[A], feed_dict=feed)
                np_loss = sess.run(loss, feed_dict=feed)
                if (i + 1) % reload == 0:
                    if (i + 1) % (reload *20) == 0:
                        t1 = datetime.datetime.now()
                        print("iteration: {} loss {} time {} lr {}".format(i, np_loss, t1 - t0, A))
                        t0 = t1
                        saver.save(sess, netFile)
                    ds = Utils.get_data(rds, rep)
                    data_gen = Utils.gen_data_batch(ds, batch_size)

            if rep > -1:
                netFile = '{}/Net_{}/PNet'.format(netFile_base, rep)
            else:
                netFile = '{}/Net/PNet'.format(netFile_base)

            saver.save(sess, netFile)
            print("Intermediate file saved at: " + netFile)


if __name__ == '__main__':
    import os
    config_file = "config7.json"
    rep = -1

    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if len(sys.argv) > 2:
        rep = int(sys.argv[2])

    process(config_file, rep)

