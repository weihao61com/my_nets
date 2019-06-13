import numpy as np
import cv2
from imagery_utils import image_resize
from pose_ana import *
import pickle
import os
from bluenotelib.common import quaternion
from bluenotelib.common.bluenote_sensor_rotation import BlueNoteSensorRotation, RotationSequence

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/..'.format(this_file_path))
from utils import Utils, PinholeCamera

lk_params = dict(winSize=(21, 21),
                 # maxLevel = 3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def featureTracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  # shape: [k,2] [k,1] [k,1]

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


_EPS = 0.0001


def rot2Euler(M):
    s2 = np.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])

    if s2 > _EPS:
        ax = np.arctan2(M[2, 1], M[2, 2])
        az = np.arctan2(M[1, 0], M[0, 0])
        if np.abs(np.sin(ax)) > 0.5:
            ay = np.arctan2(-M[2, 0], M[2, 1] / np.sin(ax))
        else:
            ay = np.arctan2(-M[2, 0], M[2, 2] / np.cos(ax))
    else:
        ax = np.arctan2(-M[1, 2], M[1, 1])
        ay = np.arctan2(-M[2, 0], s2)
        az = 0.0

    # if ax < -np.pi / 2:
    #    ax += np.pi

    return np.array([ax, ay, az]) * 180.0 / np.pi


def get_location(line):
    ss = line.strip().split()
    x = float(ss[3])
    y = float(ss[7])
    z = float(ss[11])
    return np.array([x, y, z])


class VisualOdometry2:
    def __init__(self, cam, sf, mini_match=20):
        # self.detector = cv2.xfeatures2d.SIFT_create(contrastThreshold=10)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.cam = cam
        self.feature = None
        self.id = -1
        self.sift_feature = sf
        self.mini_match = mini_match

    def get_feature(self, pose, scale=0):
        if pose.fs is None:

            try:
                # print pose.filename
                img = cv2.imread(pose.filename)
            except:
                raise Exception("failed load file {}".format(pose.filename))

            if scale > 1:
                img = image_resize(img, scale)
            fs = self.sift_feature.get_sift_feature(img)

            # fs = self.detector.detectAndCompute(img, None)
            # print pose.filename, len(fs[1])
            pose.fs = fs
        return pose.fs

    def get_match_point_2(self, curent_feature):
        matches = self.matcher.knnMatch(self.feature[1], curent_feature[1], k=2)
        pts1 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                #values = (curent_feature[0][m.trainIdx],
                #          self.feature[0][m.queryIdx],
                #          self.feature[0][n.queryIdx],
                #          m.distance, n.distance)
                values = (self.feature[0][m.queryIdx],
                          curent_feature[0][m.trainIdx],
                          curent_feature[0][n.trainIdx],
                          m.distance, n.distance)
                pts1.append(values)

        return pts1

    def get_match_point(self, curent_feature):
        matches = self.matcher.knnMatch(self.feature[1], curent_feature[1], k=2)
        pts2 = []
        pts1 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                pts2.append(self.feature[0][m.queryIdx].pt)
                pts1.append(curent_feature[0][m.trainIdx].pt)

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        return pts1, pts2

    def get_features_2(self, id1, pose1, pose2):

        #self.pose_R = np.linalg.inv(pose2.m3x3).dot(pose1.m3x3)
        #self.pose_T = pose2.tran - pose1.tran
        self.pose_R = np.linalg.inv(pose1.m3x3).dot(pose2.m3x3)
        self.pose_T = np.linalg.inv(pose1.m3x3).dot(pose2.tran - pose1.tran)

        # a1 = Utils.rotationMatrixToEulerAngles(self.pose_R)*180/np.pi
        angle = np.array(BlueNoteSensorRotation.get_rotation_angles(self.pose_R, RotationSequence.XZY))
        # a1 = BlueNoteSensorRotation.get_rotation_angles(pose1.m3x3, RotationSequence.XZY)
        # print 'detail', id1, angle[0],angle[1],angle[2],
        # a1 = pose1.tran
        # print a1[0],a1[1],a1[2],
        # a1 = pose2.tran - pose1.tran
        # print a1[0],a1[1],a1[2],np.linalg.norm(pose2.tran - pose1.tran),
        # a1 = self.pose_T
        #print a1[0], a1[1], a1[2]

        if id1 != self.id:
            self.id = id1
            self.feature = self.get_feature(pose1)

        px = self.get_match_point_2(self.get_feature(pose2))
        if len(px) > self.mini_match:
            features = []
            for p in px:
                p0 = p[0].pt
                p1 = p[1].pt
                p2 = p[2].pt
                d1 = p[3]
                d2 = d1 / p[4]
                a0 = p[0].angle/180*np.pi - np.pi
                a1 = a0 - p[1].angle/180*np.pi
                a2 = a0 - p[2].angle/180*np.pi
                s0 = p[0].size
                s1 = p[1].size
                s2 = p[2].size
                r0 = p[0].response
                r1 = p[1].response
                r2 = p[2].response
                f = [p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], d1, d2,
                     a0, a1, a2, s0, s1, s2, r0, r1, r2]
                features.append(f)

            self.features = np.array(features)
            self.truth = np.concatenate((angle, self.pose_T))
        else:
            self.features = []

    def get_features(self, id1, pose1, pose2, proc=False):

        self.pose_R = np.linalg.inv(pose1.m3x3).dot(pose2.m3x3)
        if id1 != self.id:
            self.id = id1
            self.feature = self.get_feature(pose1)

        feature = self.get_feature(pose2)
        px_new, px_last = self.get_match_point(feature)
        if px_new.shape[0] > 10:
            self.features = np.concatenate((px_new, px_last), 1)
            self.truth = Utils.rotationMatrixToEulerAngles(self.pose_R)
            if proc:
                self.matches = len(px_new)
                E, mask = cv2.findEssentialMat(px_new, px_last, cameraMatrix=self.cam.mx,
                                               method=cv2.RANSAC)  # cv2.RANSAC)LMEDS
                # , prob=0.999, threshold=10.0)
                mh, R, t, mask0 = cv2.recoverPose(E, px_new, px_last, cameraMatrix=self.cam.mx)
                m1 = np.mean(mask)
                m2 = np.mean(mask0) / 255.0

                self.inline = mh
                self.R = R
                self.t = t
                self.m1 = m1  # np.mean(mask)
                self.m2 = m2
        else:
            self.features = []

    def get_features_inline(self, id1, pose1, pose2):

        self.pose_R = np.linalg.inv(pose1.m3x3).dot(pose2.m3x3)
        if id1 != self.id:
            self.id = id1
            self.feature = self.get_feature(pose1)

        feature = self.get_feature(pose2)
        px_new, px_last = self.get_match_point(feature)
        self.truth = Utils.rotationMatrixToEulerAngles(self.pose_R)
        if px_new.shape[0] > 10:

            E, mask = cv2.findEssentialMat(px_new, px_last, cameraMatrix=self.cam.mx,
                                           method=cv2.RANSAC, prob=0.999, threshold=10.0)
            # mh, R, t, mask = cv2.recoverPose(E, px_new, px_last, cameraMatrix=self.cam.mx)
            px1 = []
            px2 = []
            for a in range(len(mask)):
                if mask[a] > 0:
                    px1.append(px_new[a])
                    px2.append(px_last[a])
            px1 = np.array(px1)
            px2 = np.array(px2)
            self.features = np.concatenate((px1, px2), 1)
        else:
            self.features = None

    def process(self, id1, pose1, id2, pose2, scale=1):

        self.pose_R = np.linalg.inv(pose1.m3x3).dot(pose2.m3x3)
        if id1 != self.id:
            self.id = id1
            self.feature = self.get_feature(pose1)

        feature = self.get_feature(pose2, scale)

        px_new, px_last = self.get_match_point(feature)
        self.matches = len(px_new)

        if len(px_new) > 10:
            # with open('tmp.p', 'w') as fp:
            #     pickle.dump((pose1.filename, pose2.filename, self.cam.mx), fp)
            # os.system('python process2.py')
            # with open('tmp.p', 'r') as fp:
            #     mh, R, t, m1, m2 =pickle.load(fp)

            # print self.matches
            E, mask = cv2.findEssentialMat(px_new, px_last, cameraMatrix=self.cam.mx,
                                           method=cv2.RANSAC)  # cv2.RANSAC)LMEDS
            # , prob=0.999, threshold=10.0)
            mh, R, t, mask0 = cv2.recoverPose(E, px_new, px_last, cameraMatrix=self.cam.mx)
            m1 = np.mean(mask)
            m2 = np.mean(mask0) / 255.0

            self.inline = mh
            self.R = R
            self.t = t
            self.m1 = m1  # np.mean(mask)
            self.m2 = m2  # np.mean(mask0)/255.0

            # p1 = []
            # p2 = []
            # for a in range(len(mask0)):
            #    if mask0[a] > 0:
            #        p1.append(px_new[a])
            #        p2.append(px_last[a])
            # self.features = np.concatenate((np.array(p1), np.array(p2)), 1)

            self.features = np.concatenate((px_new, px_last), 1)
            self.truth = Utils.rotationMatrixToEulerAngles(self.pose_R)
            self.mask1 = mask
            self.mask2 = mask0

        else:
            print("No match {}: {} {}".format(self.matches, id1, id2))


class VisualOdometry_Not:

    def __init__(self, cam, annotations):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_dR = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.mx = cam.mx
        self.matches = 0
        if type(annotations) is str:
            with open(annotations) as f:
                self.annotations = f.readlines()
        else:
            self.annotations = annotations

        self.detector = cv2.xfeatures2d.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def getAbsoluteScale(self, frame_id):  # specialized for KITTI odometry dataset
        loc_prev = get_location(self.annotations[frame_id - 1]) - self.true_loc0
        # ss = self.annotations[frame_id - 1].strip().split()
        # x_prev = float(ss[3]) - self.trueX0
        # y_prev = float(ss[7]) - self.trueY0
        # z_prev = float(ss[11]) - self.trueZ0
        locs = get_location(self.annotations[frame_id]) - self.true_loc0
        # ss = self.annotations[frame_id].strip().split()
        # x = float(ss[3]) - self.trueX0
        # y = float(ss[7]) - self.trueY0
        # z = float(ss[11]) - self.trueZ0
        self.trueX, self.trueY, self.trueZ = locs[0], locs[1], locs[2]
        return np.linalg.norm(loc_prev - locs)

    def processFirstFrame(self, frame_id):
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME
        self.true_loc0 = get_location(self.annotations[frame_id])

    def processSecondFrame(self, frame_id):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)

        # E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                       cameraMatrix=self.mx, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        mh, self.cur_dR, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                                            cameraMatrix=self.mx)
        self.matches = mh

        self.cur_R = self.cur_dR
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur
        self.true_loc = get_location(self.annotations[frame_id]) - self.true_loc0

        # ss = self.annotations[frame_id].strip().split()
        # x = float(ss[3]) - self.trueX0
        # y = float(ss[7]) - self.trueY0
        # z = float(ss[11]) - self.trueZ0
        # self.trueX, self.trueY, self.trueZ = x, y, z

    def get_match_point(self, last_frame, curent_frame):
        f1 = self.detector.detectAndCompute(last_frame, None)
        f2 = self.detector.detectAndCompute(curent_frame, None)

        matches = self.matcher.knnMatch(f1[1], f2[1], k=2)

        pts2 = []
        pts1 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                pts2.append(f2[0][m.trainIdx].pt)
                pts1.append(f1[0][m.queryIdx].pt)
        return np.array(pts1), np.array(pts2)

    def processFrame(self, frame_id):
        px_last, px_new = self.get_match_point(self.last_frame, self.new_frame)

        E, mask = cv2.findEssentialMat(px_new, px_last, cameraMatrix=self.mx,
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        mh, self.cur_dR, t, mask = cv2.recoverPose(E, px_new, px_last, cameraMatrix=self.mx)
        self.matches = mh

        absolute_scale = self.getAbsoluteScale(frame_id)
        if absolute_scale > 0.1:
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = self.cur_dR.dot(self.cur_R)
        if (self.px_ref.shape[0] < kMinNumFeature):
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        assert (img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[
            1] == self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        if (self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
        elif (self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame(frame_id)
        elif (self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame(frame_id)
        self.last_frame = self.new_frame
