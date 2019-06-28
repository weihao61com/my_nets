import sys
import os
import cv2
import pickle

this_file_path = os.path.dirname(os.path.realpath(__file__))
HOME = '{}/../..'.format(this_file_path)
sys.path.append('{}/my_nets'.format(HOME))

from image_pairing.pose_ana import \
    load_kitti_poses, load_indoor_7_poses, load_TUM_poses
from image_pairing.cv_location import VisualOdometry2
from image_pairing.imagery_utils import SiftFeature
import datetime


def ModifiedKeyPoint(f):
    # pt, angle, size, response, class_id, octave
    return (f.pt[0], f.pt[1], f.angle, f.size, f.response, f.class_id, f.octave)

class RAS_D:
    def __init__(self):
        self.poses = None
        self.cam = None
        self.features = {}
        self.sf = SiftFeature()
        self.matches = {}

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def set_poses(self, poses, cam):
        self.poses = poses
        self.cam = cam

    def get_feature(self, pose):
        try:
            img = cv2.imread(pose.filename)
        except:
            raise Exception("failed load file {}".format(pose.filename))

        return self.sf.get_sift_feature(img)

    def modify_features(self):
        for seq in self.features:
            features = self.features[seq]
            A = {}
            for img_id in features:
                fs = features[img_id]
                keypoints = []
                for f in fs[0]:
                    print f
                    keypoints.append(ModifiedKeyPoint(f))
                A[img_id] = [keypoints, fs[1]]
            self.features[seq] = A

    def process(self, range2):
        range3 = 0
        range1 = -range2

        length = 0
        nt = 0
        t0 = datetime.datetime.now()

        for seq in self.poses:
            poses = poses_dic[seq]
            print seq, len(poses)
            self.features[seq] = {}
            self.matches[seq] = {}
            for id in poses:
                self.features[seq][id] = self.get_feature(poses[id])
                if len(self.features[seq])%200==0:
                    print seq, len(self.features[seq]), datetime.datetime.now()-t0

            features = self.features[seq]
            print 'matching', datetime.datetime.now()-t0
            t0 = datetime.datetime.now()

            for id1 in poses:
                for id2 in poses:
                    if abs(id2 - id1) <= range3:
                        continue

                    if range1 <= id2 - id1 <= range2:
                        pts1 = []

                        f1 = features[id1]
                        f2 = features[id2]
                        matches = self.matcher.knnMatch(f1[1], f2[1], k=2)
                        for m,n in matches:
                            if m.distance < 0.8 * n.distance:
                                values = (m.queryIdx, m.trainIdx, n.trainIdx,
                                          m.distance, n.distance)
                                pts1.append(values)

                        #
                        #
                        # # ratio test as per Lowe's paper
                        # for i, (m, n) in enumerate(matches):
                        #     if m.distance < 0.8 * n.distance:
                        #         # values = (curent_feature[0][m.trainIdx],
                        #         #          self.feature[0][m.queryIdx],
                        #         #          self.feature[0][n.queryIdx],
                        #         #          m.distance, n.distance)
                        #         values = (f1[0][m.queryIdx],
                        #                   f2[0][m.trainIdx],
                        #                   f2[0][n.trainIdx],
                        #                   m.distance, n.distance)
                        #         pts1.append(values)

                        length += len(pts1)
                        nt += 1

                        if nt % 1000 == 0:
                            print nt, id1, datetime.datetime.now() - t0, length/nt, length
                            t0 = datetime.datetime.now()
                        self.matches[seq][(id1, id2)] = pts1
        print
        print nt, datetime.datetime.now() - t0, length/nt

if __name__ == '__main__':
    range2 = 2

    key = 'heads'
    mode = 'Test'
    #key = 'rgbd_dataset_freiburg3_long_office_household'
    #mode = 'Train'

    if len(sys.argv)>1:
        key = sys.argv[1]
    if len(sys.argv)>2:
        mode = sys.argv[2]

    print key, mode

        # location = '/home/weihao/Projects/cambridge/OldHospital'
        # pose_file = 'dataset_train.txt'
        # poses_dic, cam = load_cambridge_poses(location, pose_file)

    if key.startswith('0'):
        location = '{}/datasets/kitti'.format(HOME)
        poses_dic, cam = load_kitti_poses(location, key + ".txt")
        key = 'kitti_{}'.format(key)
    elif key.startswith('rgbd'):
        location = '{}/datasets/TUM'.format(HOME)
        poses_dic, cam = load_TUM_poses(location, key)
    else:
        location = "{}/datasets/indoors/{}".format(HOME, key)  # office" #heads
        poses_dic, cam = load_indoor_7_poses(location, "{}Split.txt".format(mode))

    filename = '{}/p_files/{}_{}_ras_s{}_3.p'.format(HOME, key, mode, range2)
    output_file = '{}/tmp/{}_{}.csv'.format(HOME, key, mode)
    print location
    print filename
    print output_file

    rasd = RAS_D()
    rasd.set_poses(poses_dic, cam)
    rasd.process(range2)
    rasd.modify_features()

    with open(filename, 'w') as fp:
        pickle.dump(rasd, fp)


