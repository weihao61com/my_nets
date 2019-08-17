import numpy as np
#from bluenotelib.common.quaternion import Quaternion
#from bluenotelib.common.coordinate_transforms import CoordinateTransforms
#from bluenotelib.common.bluenote_sensor_rotation import BlueNoteSensorRotation, RotationSequence
from sortedcontainers import SortedDict
import os
import cv2
import math
import evo.core.transformations as tr


def quaternion_matrix(quaternion):
    _EPS = np.finfo(float).eps * 4.0

    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

class SiftFeature:
    def __init__(self, target=2000):
        self.th=0.01
        self.target = target

    def get_sift_feature(self, img):
        th_low = 0.01
        target_length = self.target
        th = self.th

        detector = cv2.xfeatures2d.SIFT_create(contrastThreshold=th)
        feat = detector.detectAndCompute(img, None)
        pre_f = feat
        if len(feat[1])>target_length:
            while len(feat[1])>target_length:
                self.th = th
                th *= 1.02
                detector = cv2.xfeatures2d.SIFT_create(contrastThreshold=th)
                pre_f = feat
                feat = detector.detectAndCompute(img, None)
        else:
            while len(feat[1])<target_length and th>th_low:
                self.th = th
                th *= 0.98
                detector = cv2.xfeatures2d.SIFT_create(contrastThreshold=th)
                pre_f = feat
                feat = detector.detectAndCompute(img, None)
        return pre_f


def image_resize(img, scale = 1.01):
    sz = img.shape
    new_size = (int(scale*sz[1]), int(scale*sz[0]))
    img = cv2.resize(img, new_size)
    x0 = (new_size[1]-sz[0])/2
    y0 = (new_size[0]-sz[1])/2
    x1 = x0 + sz[0]
    y1 = y0 + sz[1]
    return img[x0:x1, y0:y1, :]

def re_normalize(m3):
    inv = np.linalg.inv(m3)
    tra = m3.transpose()
    m1 = (inv+tra)/2
    m2 = np.linalg.inv(m1)
    A = tr.euler_from_matrix(m2)
    m = tr.euler_matrix(A[0], A[1], A[2])
    return m[:3, :3]

class Pose:
    def __init__(self, line, location, pose_file, data=0, id=0):
        strs = line[:-1].split()
        self.fs = None
        if data==0: # kitti
            a = np.reshape(np.array(list(map(float, strs))), (3, 4))
            nm = '{0}/sequences/{1}/image_1/{2}.png'.format(location, pose_file, str(id).zfill(6))
            self.filename = nm
            self.m3x3 = a[:, :3]
            self.tran = a[:, 3]
            self.Q4 = np.concatenate((a, np.array([[0,0,0,1]])))

            #q = Quaternion(m3x3=self.m3x3)
        elif data==1: # cambridge
            # mx = np.array([[0,1,0], [0,0,1], [-1,0,0]])
            # mx = np.array([[1,0,0], [0,1,0], [0,0,1]])

            self.filename = os.path.join(location, pose_file)
            a = location
            self.tran = a[:3]
            M = tr.quaternion_matrix(a[3:])
            M[:3, 3] = self.tran
            self.Q4 = M
        elif data==2: # microsoft indoor 7
            self.filename = pose_file
            p_file = pose_file[:-9] + 'pose.txt'
            m3x4 = np.loadtxt(p_file)
            # m3 = re_normalize(m3x4[:3, :3])
            m3 = (m3x4[:3, :3])
            self.m3x3 = m3
            self.tran = m3x4[:3, 3]
            basename = os.path.basename(pose_file)[:-10]
            self.id = int(basename[-6:])
            m3x4[:3, :3] = m3
            self.Q4 = m3x4
        elif data ==3: #TUM????
            self.filename = os.path.join(line, pose_file)
            a = location
            tran = a[:3]
            quat = a[3:]  # n x 4
            quat = np.roll(quat, 1, axis=0)
            self.tran = tran
            self.m3x3 = tr.quaternion_matrix(quat)[:3,:3]
            Q4 = np.identity(4)
            Q4[:3, :3] = self.m3x3
            Q4[:3, 3] = self.tran
            self.Q4 = Q4

    def get_direction(self, pose, cam):
        return np.linalg.inv(pose.m3x3).dot(self.m3x3)

    def get_tran(self, pose):
        return self.tran - pose.tran

    def get_string(self):
        return 'a b c {} d e f {} g h i {}'.\
            format(self.tran[0], self.tran[1], self.tran[2])

def pose_realign(poses):
    mx = None
    new_poses = SortedDict()
    for id in poses:
        if mx is None:
            mx = np.linalg.inv(poses[id].m3x3)
        q = poses[id]
        q.tran = mx.dot(q.tran)
        q.m3x3 = mx.dot(q.m3x3)
        new_poses[id] = q
        #if len(new_poses)>50:
        #    break
    return new_poses


