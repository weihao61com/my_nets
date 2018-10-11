import cv2
import numpy as np
import pickle
from utils import SiftFeature


def get_match_point(curent_feature, feature):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(feature[1], curent_feature[1], k=2)

    pts2 = []
    pts1 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(feature[0][m.queryIdx].pt)
            pts1.append(curent_feature[0][m.trainIdx].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    return pts1, pts2


if __name__ == '__main__':
    with open('tmp.p', 'r') as fp:
        file1, file2, mx = pickle.load(fp)

    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)

    sf = SiftFeature()

    f1 = sf.get_sift_feature(img1)
    f2 = sf.get_sift_feature(img2)

    px_new, px_last = get_match_point(f1, f2)

    # print self.matches
    E, mask = cv2.findEssentialMat(px_new, px_last, cameraMatrix=mx,
                                   method= cv2.RANSAC) # cv2.RANSAC)
    # , prob=0.999, threshold=10.0)
    mh, R, t, mask0 = cv2.recoverPose(E, px_new, px_last, cameraMatrix=mx)

    m1 = np.mean(mask)
    m2 = np.mean(mask0)/255.0

    with open('tmp.p', 'w') as fp:
        pickle.dump((mh, R, t, m1, m2), fp)
