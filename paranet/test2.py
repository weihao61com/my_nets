# Import the converted model's class
import numpy as np
import os
import tensorflow as tf
from posenet import GoogLeNet as PoseNet
import glob
import math
from utils import Utils


def process(ds, netFile, net, image, shift):
    if len(ds.images) == 0:
        return

    results = np.zeros((len(ds.images), 2))
    p3_x = net.layers['cls3_fc_pose_xyz']
    p3_q = net.layers['cls3_fc_pose_wpqr']

    init = tf.global_variables_initializer()  # initialize_all_variables()
    saver = tf.train.Saver()

    # config = tf.ConfigProto(
    #    device_count = {'GPU': -1},
    #    inter_op_parallelism_threads=3,
    #    intra_op_parallelism_threads=3
    # )

    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': -1},
        allow_soft_placement=True,
        log_device_placement=False,
        inter_op_parallelism_threads=3,
        intra_op_parallelism_threads=3
    )

    with tf.Session(config=session_conf) as sess:

        # Load the data
        sess.run(init)
        saver.restore(sess, netFile)
        data_raw = []
        nt = 0
        for i in ds.images:
            np_image = ds.images[i]
            np_xy = ds.xys[i]
            xy = np_xy.reshape((1, 1, 1, len(np_xy)))

            feed = {image: np_image, shift: xy}

            pose_q = np.asarray(ds.poses[i][3:7])
            pose_x = np.asarray(ds.poses[i][0:3])
            predicted_x, predicted_q = sess.run([p3_x, p3_q], feed_dict=feed)

            pose_q = np.squeeze(pose_q)
            pose_x = np.squeeze(pose_x)
            predicted_q = np.squeeze(predicted_q)
            predicted_x = np.squeeze(predicted_x)

            data_raw.append((pose_q, pose_x, predicted_q, predicted_x))

            # Compute Individual Sample Error
            q1 = pose_q / np.linalg.norm(pose_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)
            d = abs(np.sum(np.multiply(q1, q2)))
            if d > 1:
                d = 1
            if d < -1:
                d = -1
            theta = 2 * np.arccos(d) * 180 / math.pi
            error_x = np.linalg.norm(pose_x - predicted_x)
            results[nt, :] = [error_x, theta]
            nt += 1
            # print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta

    median_result = np.median(results, axis=0)
    print 'Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'


def main():
    import sys
    config_file = "config.json"

    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    rep = None
    if len(sys.argv) > 2:
        rep = int(sys.argv[2])

    test_data = True
    if len(sys.argv) > 3:
        test_data = int(sys.argv[3])==0

    js = Utils.load_json_file(config_file)
    location = js['directory']
    if test_data:
        dataset = js['testing_dataset']
    else:
        dataset = js['training_dataset']

    netFile = js['netFile']
    if rep is None:
        rep = int(js['rep'])
    if rep > -1:
        netFile = '{}/Net_{}/PNet'.format(netFile, rep)
    else:
        netFile = '{}/Net/PNet'.format(netFile)

    if len(sys.argv) > 4:
        netFile = sys.argv[4]

    classes = glob.glob(os.path.join(location, '*'))
    num_class = len(classes)

    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    shift = tf.placeholder(tf.float32, [1, 1, 1, num_class])

    net = PoseNet({'data': images, 'shift': shift})

    for a in range(num_class):
        if rep==-1 or rep==a:
            rds = Utils.get_raw_data_indoor(location, dataset, a)
            ds = Utils.get_data(rds, a, 2.0, ss=480)

            process(ds, netFile, net, images, shift)


if __name__ == '__main__':
    import sys

    main()
