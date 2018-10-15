import cv2
from matplotlib import pyplot as plt
import numpy as np
#from bluenotelib.common.quaternion import Quaternion
#from bluenotelib.common.bluenote_sensor_rotation import BlueNoteSensorRotation, RotationSequence
#from sortedcontainers import SortedDict

def rotation_to_rph(R):
    pitch, roll, heading = BlueNoteSensorRotation.get_rotation_angles(R, RotationSequence.XYZ)
    return np.array([pitch, roll, heading])

from gms_matcher import GmsMatcher

class FeatureMatching():

    def __init__(self, det = 'sift'):
        self.detector = None
        self.det = det
        self.size = None

        if det=='sift_flann':
            self.detector = cv2.xfeatures2d.SIFT_create()
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        elif det=='orb_gms':
            self.detector = cv2.ORB_create(10000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        elif det=='sift_bf':
            self.detector = cv2.xfeatures2d.SIFT_create()
            self.matcher = cv2.BFMatcher()

        elif det=='orb_org':
            orb = cv2.ORB_create(5000)
            orb.setFastThreshold(0)
            if cv2.__version__.startswith('3'):
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            else:
                matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
            gms = GmsMatcher(orb, matcher)
            self.detector = GmsMatcher(orb, matcher)
            self.matcher = GmsMatcher(orb, matcher)

        else:
            raise Exception('Unknown detector {}'.format(det))

    def dc(self, img):
        return self.detector.detectAndCompute(img, None)


    def matching(self, f1, f2, img1, img2):
        if self.det=='sift_flann':
            matches = self.matcher.knnMatch(f1[1], f2[1], k=2)

            pts2 = []
            pts1 = []
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.8 * n.distance:
                    pts2.append(f2[0][m.trainIdx].pt)
                    pts1.append(f1[0][m.queryIdx].pt)
            return pts1, pts2

        elif self.det=='orb_gms':
            matches = self.matcher.match(f1[1], f2[1])
            matches = cv2.xfeatures2d.matchGMS(size1=self.size,
                                               size2=self.size,
                                               keypoints1=f1[0],
                                               keypoints2=f2[0],
                                               matches1to2=matches,
                                               withRotation=False,
                                               withScale=True,
                                               thresholdFactor=6.0)
            pts1 = np.array([f1[0][m.queryIdx].pt for m in matches])
            pts2 = np.array([f2[0][m.trainIdx].pt for m in matches])
            return pts1, pts2

        elif self.det=='sift_bf':
            matches = self.matcher.match(f1[1], f2[1])
            pts1 = np.array([f1[0][m.queryIdx].pt for m in matches])
            pts2 = np.array([f2[0][m.trainIdx].pt for m in matches])
            return pts1, pts2

        elif self.det=='orb_org':
            matches = self.detector.compute_matches(img1, img2)
            pts1 = np.array([self.detector.keypoints_image1[m.queryIdx].pt for m in matches])
            pts2 = np.array([self.detector.keypoints_image2[m.trainIdx].pt for m in matches])
            return pts1, pts2

        else:
            raise Exception('Unknown detector {}'.format(self.det))

    def set_image_size(self, sz):
        self.size = sz


def matching(img_nms, alg, step, focal_length):

    rots = []
    fm = FeatureMatching(alg)

    num_matches = 0

    for a in range(1, len(img_nms)):
        img1 = cv2.imread(img_nms[a][0])
        img_features1 = fm.dc(img1) #sift.detectAndCompute(img1, None)
        pose1 = img_nms[a][2]
        tran1 = img_nms[a][1]
        fm.set_image_size(img1.shape[:2])

        if a+step in img_nms:
            img2 = cv2.imread(img_nms[a+step][0])
            img_features2 = fm.dc(img2)  # sift.detectAndCompute(img1, None)
            pose2 = img_nms[a+step][2]
            tran2 = img_nms[a+step][1]

            pts1, pts2 = fm.matching(img_features1, img_features2, img1, img2)

            pts1 = np.array(pts1)
            pts2 = np.array(pts2)
            idxs = np.arange(0, len(pts1))
            np.random.shuffle(idxs)
            pts1 = pts1[idxs]
            pts2 = pts2[idxs]
            num_matches += len(pts1)

            #focal_length = 1500 #3225.6/3024.0*960.0
            principal_point = tuple((np.array(img1.shape[0:2]) / 2.0).astype(int))
            mx = np.array([[focal_length, 0, principal_point[0]],
                          [0, focal_length, principal_point[1]],
                          [0,0,1]])

            E, mask = cv2.findEssentialMat(pts1, pts2,  cameraMatrix=mx, method=cv2.RANSAC, prob=0.999, threshold=3.0)
            #R1, R2, t0 = cv2.decomposeEssentialMat(E)
            retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=mx) #pp=principal_point
            rot = rotation_to_rph(R)
            d = np.linalg.inv(pose2).dot(pose1)
            dr = rotation_to_rph(d)
            print('{} {} {} {} {} {} {} {} {}'.format(a, len(pts1), rot[0], rot[1], rot[2], dr[0], dr[1], dr[2],
                                        np.linalg.norm(tran1-tran2)))
            rots.append(rot)
        if a>1000:
            break

    return rots, num_matches/(len(img_nms)-1)


def load_poses(location, pose_file):
    poses = SortedDict()
    skip = 2
    filename = os.path.join(location, pose_file)
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            # print line[:-1]
            if skip<0:
                strs = line[:-1].split()
                id = int(strs[0][10:15])
                filename = os.path.join(location, strs[0])
                a = np.array(map(float, strs[1:]))
                q = Quaternion(qw=a[3], qx=a[4], qy=a[5], qz=a[6])
                rot = Quaternion.to_rotation_matrix(q)
                poses[id] = (filename, a[:3], rot)
            skip -= 1

    return poses, 1500


def load_kitty_poses(location, pose_file):
    poses = SortedDict()
    filename = '{}/poses/{}.txt'.format(location, pose_file)
    id = 0
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            strs = line[:-1].split()
            a = np.reshape(np.array(map(float, strs)), (3,4))
            nm = '{0}/sequences/{1}/image_0/{2}.png'.format(location, pose_file, str(id).zfill(6))
            q = Quaternion(m3x3=a[:3])
            # poses[id] = (nm, np.array(a[:, 3]), np.array(a[:, :3]), q)
            poses[id] = (nm, a[:, 3], a[:, :3], q)
            id += 1

    return poses, 1000

import datetime
import os
import sys
# location = '/Users/weihao/Downloads/ShopFacade'
# pose_file = 'dataset_train.txt'
# poses = load_poses(location, pose_file)

location = '/Users/weihao/BlueNoteData/dataset'
pose_file = '00'
poses, focal = load_kitty_poses(location, pose_file)

print('Total images {}'.format(len(poses)))

algs = ['orb_org', 'orb_gms', 'sift_flann', 'sift_bf']
algs = ['sift_flann']

for alg in algs:
    t0 = datetime.datetime.now()
    rots, nm = matching(poses, alg, step=2, focal_length=focal)
    # rots = np.array(rots)
    # print('{:10s}: \tmatches= {}, mean= {}, std={}: \t{}'.format(alg, nm, np.mean(rots, 0), np.std(rots, 0), datetime.datetime.now()-t0))