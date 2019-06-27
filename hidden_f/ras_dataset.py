import sys
import os
import cv2

this_file_path = os.path.dirname(os.path.realpath(__file__))
HOME = '{}/../..'.format(this_file_path)
sys.path.append('{}/my_nets'.format(HOME))

from image_pairing.pose_ana import \
    load_kitti_poses, load_indoor_7_poses, load_TUM_poses
from image_pairing.cv_location import VisualOdometry2
from image_pairing.imagery_utils import SiftFeature
import datetime


class RAS_D:
    def __init__(self):
        self.poses = None
        self.cam = None
        self.features = {}
        self.sf = SiftFeature()

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
            for id in poses:
                self.features[seq][id] = self.get_feature(poses[id])

            features = self.features[seq]

            for id1 in poses:
                for id2 in poses:
                    if abs(id2 - id1) <= range3:
                        continue
                    if id2<=id1:
                        continue

                    if range1 <= id2 - id1 <= range2:
                        pts1 = []

                        f1 = features[id1]
                        f2 = features[id2]
                        matches = self.matcher.knnMatch(f1[1], f2[1], k=2)
                        # ratio test as per Lowe's paper
                        for i, (m, n) in enumerate(matches):
                            if m.distance < 0.8 * n.distance:
                                # values = (curent_feature[0][m.trainIdx],
                                #          self.feature[0][m.queryIdx],
                                #          self.feature[0][n.queryIdx],
                                #          m.distance, n.distance)
                                values = (f1[0][m.queryIdx],
                                          f2[0][m.trainIdx],
                                          f2[0][n.trainIdx],
                                          m.distance, n.distance)
                                pts1.append(values)

                        length += len(pts1)
                        nt += 1

                        if id1 % 100 == 0:
                            print nt, id1, datetime.datetime.now() - t0, length/nt
                            t0 = datetime.datetime.now()

        print nt, datetime.datetime.now() - t0, length/nt

if __name__ == '__main__':
    range2 = 1

    key = 'stairs'
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

    filename = '{}/p_files/{}_{}_cv_s{}_3.p'.format(HOME, key, mode, range2)
    output_file = '{}/tmp/{}_{}.csv'.format(HOME, key, mode)
    print location
    print filename
    print output_file

    rasd = RAS_D()
    rasd.set_poses(poses_dic, cam)
    rasd.process(range2)





