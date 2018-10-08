# Import the converted model's class
import numpy as np
import tensorflow as tf
from posenet import GoogLeNet as PoseNet
import math
from utils import Utils

def process(ds, netFile, net, image, shift):
    results = np.zeros((len(ds.images), 2))
    p3_x = net.layers['cls3_fc_pose_xyz']
    p3_q = net.layers['cls3_fc_pose_wpqr']

    init = tf.global_variables_initializer()  # initialize_all_variables()
    saver = tf.train.Saver()

    #config = tf.ConfigProto(
    #    device_count = {'GPU': -1},
    #    inter_op_parallelism_threads=3,
    #    intra_op_parallelism_threads=3
    #)

    session_conf = tf.ConfigProto(
        device_count={'CPU' : 1, 'GPU' : -1},
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
        for i in range(len(ds.images)):
            np_image = ds.images[i]
            np_xy = ds.xys[i]
            xy = np_xy.reshape((1,1, 1, 4))

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
            results[i, :] = [error_x, theta]
            # print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta

    median_result = np.median(results, axis=0)
    print 'Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'


def main():
    import sys
    config_file = "/home/weihao/posenet/paranet/config.json"

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    js = Utils.load_json_file(config_file)
    location = js['directory']
    tr_dataset = js['training_dataset']
    dataset = js['testing_dataset']
    netFile = js['netFile']

    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    shift = tf.placeholder(tf.float32, [1, 1, 1, 4])

    net = PoseNet({'data': images, 'shift': shift})

    ds = []
    rds = Utils.get_raw_data(location, dataset)
    for a in range(4):
        ds.append(Utils.get_data(rds, a))
    #rds = Utils.get_raw_data(location, tr_dataset)
    #ds.append(Utils.get_data(rds, 1, False))

    for a in range(len(ds)):
        process(ds[a], netFile, net, images, shift)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    main()
