import sys
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/..'.format(this_file_path))
from utils import Utils

import cv2
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
from imagery_utils import SiftFeature, image_resize



def draw_matching(img_name_pair, scale):
    sf = SiftFeature(4000)
    print img_name_pair[0]
    img0 = cv2.imread(img_name_pair[0])
    feat0 = sf.get_sift_feature(img0)
    print img_name_pair[1]
    img1 = cv2.imread(img_name_pair[1])
    if scale>1:
        img1 = image_resize(img1, scale=scale)
    feat1 = sf.get_sift_feature(img1)

    sz = img0.shape
    focal = 525

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matcher.knnMatch(feat0[1], feat1[1], k=2)
    print 'matches', len(matches), 'out of', len(feat0[1]), len(feat1[1])
    matchesMask = [[0, 0] for i in xrange(len(matches))]
    nt = 0
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchesMask[i] = [1, 0]
            nt += 1

    #print 'matches:', nt, 'from', len(feat0[1]),len(feat1[1])

    draw_params = dict(singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    out_img = cv2.drawMatchesKnn(img0, feat0[0], img1, feat1[0], matches, None, **draw_params)
    plt.imshow(out_img, interpolation='none')
    plt.show()

    mx = np.array([[focal, 0, sz[1]/2], [0, focal, sz[0]/2], [0, 0, 1]])

    pts2 = []
    pts1 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(feat0[0][m.queryIdx].pt)
            pts1.append(feat1[0][m.trainIdx].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    #for p in range(len(pts1)):
    #    print pts1[p][0], pts1[p][1], pts2[p][0], pts2[p][1]
    print 'Good Match', len(pts1), 'out of', len(matches)
    print mx
    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=mx,
                                   method = cv2.RANSAC, prob = 0.999, threshold = 10.0)
    #print E
    #print mx
    mh, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=mx)
    print mh
    print R
    print np.reshape(t,(3))
    print Utils.rotationMatrixToEulerAngles(R)*180/3.1416, \
        Utils.rotationMatrixToEulerAngles(R)


import sys
scale = 1.0
if len(sys.argv)>1:
    scale = float(sys.argv[1])

#img_name_pair = ('/Users/weihao/Projects/posenet/data/OldHospital/seq1/frame00001.png',
#                 '/Users/weihao/Projects/posenet/data/OldHospital/seq1/frame00002.png')
#img_name_pair = ('/Users/weihao/PY/I7/office/seq-05/frame-000273.color.png',
#                 '/Users/weihao/PY/I7/office/seq-05/frame-000272.color.png')
img_name_pair = ('/home/weihao/Projects/datasets/indoors/heads/seq-01/frame-000440.color.png',
                 '/home/weihao/Projects/datasets/indoors/heads/seq-01/frame-000425.color.png')
img_name_pair = ('/home/weihao/Projects/colmap_features/heads_Test/images/1_frame-000000.color.png',
                 '/home/weihao/Projects/colmap_features/heads_Test/images/1_frame-000029.color.png')
#img_name_pair = ('/home/weihao/Projects/cambridge/OldHospital/seq1/frame00001.png',
#                 '/home/weihao/Projects/cambridge/OldHospital/seq1/frame00002.png')
draw_matching(img_name_pair, scale)
