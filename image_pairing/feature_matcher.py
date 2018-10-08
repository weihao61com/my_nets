import cv2
import numpy as np
from utils import get_sift_feature


class FeatureMatcher:

    def __init__(self):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def get_matching_features(self, img1, img2, feat1=None, feat2=None):
        if feat1 is None:
            feat1 = get_sift_feature(img1)
        if feat2 is None:
            feat2 = get_sift_feature(img2)

        matches = self.matcher.knnMatch(feat1[1], feat2[1], k=2)

        pts2 = []
        pts1 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                pts2.append(feat1[0][m.queryIdx].pt)
                pts1.append(feat2[0][m.trainIdx].pt)

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        return np.concatenate((pts1, pts2), 1)

img_name_pair = ('/Users/weihao/PY/I7/office/seq-05/frame-000194.color.png',
                 '/Users/weihao/PY/I7/office/seq-05/frame-000195.color.png')
matcher = FeatureMatcher()

img0 = cv2.imread(img_name_pair[0])
img1 = cv2.imread(img_name_pair[1])

# mx = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
mx = np.array([[600, 0, 240], [0, 600, 320], [0, 0, 1]])

features = matcher.get_matching_features(img0, img1)
E, mask = cv2.findEssentialMat(features[:, 0:2], features[:, 2:4], cameraMatrix=mx,
                               method=cv2.RANSAC, prob=0.00999, threshold=10.0)
# print E
# print mx
mh, R, t, mask = cv2.recoverPose(E, features[:, 0:2], features[:, 2:4], cameraMatrix=mx)
print mh
print R
print np.reshape(t, (3))
