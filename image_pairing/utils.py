import numpy as np
from bluenotelib.common.quaternion import Quaternion
from bluenotelib.common.coordinate_transforms import CoordinateTransforms
from bluenotelib.common.bluenote_sensor_rotation import BlueNoteSensorRotation, RotationSequence
from sortedcontainers import SortedDict
import os
import cv2

class SiftFeature:
    def __init__(self, target=2000):
        self.th=0.5
        self.target = target

    def get_sift_feature(self, img):
        th_low = 0.1
        target_length = self.target
        th = self.th

        detector = cv2.xfeatures2d.SIFT_create(edgeThreshold=th)
        feat = detector.detectAndCompute(img, None)
        pre_f = feat
        if len(feat[1])>target_length:
            while len(feat[1])>target_length:
                self.th = th
                th *= 1.1
                detector = cv2.xfeatures2d.SIFT_create(edgeThreshold=th)
                pre_f = feat
                feat = detector.detectAndCompute(img, None)
        else:
            while len(feat[1])<target_length and th>th_low:
                self.th = th
                th *= 0.9
                detector = cv2.xfeatures2d.SIFT_create(edgeThreshold=th)
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


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

class Pose:
    def __init__(self, line, location, pose_file, data=0, id=0):
        strs = line[:-1].split()
        if data==0:
            a = np.reshape(np.array(map(float, strs)), (3, 4))
            nm = '{0}/sequences/{1}/image_1/{2}.png'.format(location, pose_file, str(id).zfill(6))
            self.filename = nm
            self.m3x3 = a[:, :3]
            self.tran = a[:, 3]
            q = Quaternion(m3x3=self.m3x3)
        elif data==1:
            # mx = np.array([[0,1,0], [0,0,1], [-1,0,0]])
            # mx = np.array([[1,0,0], [0,1,0], [0,0,1]])

            self.filename = '{}/{}'.format(location, strs[0])
            a = np.array(map(float, strs[1:]))
            self.tran = a[:3]
            q = Quaternion(qw=a[3], qx=a[4], qy=a[5], qz=a[6])
            self.m3x3 = Quaternion.to_rotation_matrix(q)
        elif data==2: # microsoft indoor 7
            self.filename = pose_file
            p_file = pose_file[:-9] + 'pose.txt'
            m3x4 = np.loadtxt(p_file)
            self.m3x3 = m3x4[:3, :3]
            self.tran = m3x4[:3, 3]
            basename = os.path.basename(pose_file)[:-10]
            self.id = int(basename[-6:])

    def get_direction(self, pose, cam):
        return np.linalg.inv(pose.m3x3).dot(self.m3x3)

    def get_tran(self, pose):
        return self.tran - pose.tran

    def get_string(self):
        return 'a b c {} d e f {} g h i {}'.\
            format(self.tran[0], self.tran[1], self.tran[2])


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]
        self.mx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])




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


