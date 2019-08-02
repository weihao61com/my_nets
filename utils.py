import cv2
import numpy as np
import os
import random
import json
from sortedcontainers import SortedDict
import sys
import pickle
import shutil
import csv
import evo.core.transformations as tr
# from bluenotelib.common.bluenote_sensor_rotation \
#    import BlueNoteSensorRotation, RotationSequence


HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'


class Config:
    def __init__(self, config_file):
        self.js = Utils.load_json_file(config_file)
        self.net_type = 'fc'
        self.af = 'af'
        self.renetFile = None
        self.att = None

        for str in self.js:
            setattr(self, str, self.js[str])

        self.tr_data = []
        self.te_data = []
        self.tr2_data = []
        self.te2_data = []
        self.nodes = []
        self.fc_nodes = []
        for key in self.js:
            if key.startswith('tr'):
                self.tr_data.append(HOME + self.js[key])
            if key.startswith('te'):
                self.te_data.append(HOME + self.js[key])
            if key.startswith('2tr'):
                self.tr2_data.append(HOME + self.js[key])
            if key.startswith('2te'):
                self.te2_data.append(HOME + self.js[key])
            if key.startswith('nodes'):
                print(key, self.js[key])
                self.nodes.append(list(map(int, self.js[key].split(','))))
            if key.startswith('fc_nodes'):
                self.fc_nodes.append(map(int, self.js[key].split(',')))

        self.netFile = HOME + 'NNs/' + self.netFile + '/fc'
        # self.netTest = fc_const.HOME + 'NNs/' + self.netTest + '/fc'
        if self.renetFile is not None:
            self.renetFile = HOME + 'NNs/' + self.renetFile + '/fc'

    def get_data(self, str, dv=None):
        return self.js[str] if str in self.js else dv


class datasource(object):

    def __init__(self, images, poses, c):
        self.images = images
        self.poses = poses
        self.xys = c
        # print("Total images {}".format(len(images)))


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


class Utils:

    @staticmethod
    def read_csv(filename, delimiter=','):
        data = []
        with open(filename, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=delimiter)
            for row in csv_reader:
                if 'latitude' in row:
                    print('\t\t\t', row)
                else:
                    data.append(row)
        # print 'Total csv data', len(data)
        return data

    @staticmethod
    def get_numbers(str):
        strs = str.split(',')
        output = []
        for s in strs:
            if '-' in s:
                num = s.split('-')
                n1 = int(num[0])
                n2 = int(num[1])
                for a in range(n1, n2 + 1):
                    output.append(a)
            else:
                output.append(int(s))
        return output


    @staticmethod
    def get_relative(p1, p2):
        Q4 = np.linalg.inv(p1.Q4).dot(p2.Q4)
        A = Utils.get_A(Q4)
        return A

    @staticmethod
    def rotationMatrixToEulerAngles(R):
        raise Exception()
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
        #return np.array([(x+z)/2, y, (x-z)/2])

    @staticmethod
    def calculate_loss(v1, v2):
        diff = v1- v2
        r = np.linalg.norm(diff, axis=1)
        loss = np.linalg.norm(r)
        return loss*loss/len(r), np.median(r)

    @staticmethod
    def calculate_stack_loss_ras(v1, v2, abs_err):
        diff = v1- v2
        r = np.linalg.norm(diff, axis=1)
        if abs_err==0:
            loss = np.mean(r*r)
        else:
            loss = np.mean(r)

        return loss, np.median(r)

    @staticmethod
    def calculate_stack_loss_avg(v1, v2, abs_err):
        if len(v2.shape) == 1:
            v2 = v2.reshape((len(v2), 1))
        L = []
        M = []
        for a in range(v1.shape[1]):
            diff = v1[:, a, :] - v2
            r = np.linalg.norm(diff, axis=1)
            if abs_err == 0:
                loss = np.mean(r * r)
            else:
                loss = np.mean(r)
            L.append(loss)
            M.append(np.median(r))
        return L, M

    @staticmethod
    def calculate_stack_loss(v1, v2):
        L = []
        M = []
        for a in range(len(v1)):
            diff = v1[a]- v2
            r = np.linalg.norm(diff, axis=1)
            loss = np.linalg.norm(r)
            L.append(loss*loss/len(r))
            M.append(np.median(r))
        return L, M

    @staticmethod
    def load_json_file(filename, verbose = True):
        out = SortedDict()
        with open(filename, 'r') as fp:
            js = json.load(fp)
            for key in js:
                if not key.startswith('_'):
                    out[key] = js[key]
            if verbose:
                for key in out:
                    print("Config *****   {}: {}".format(key, out[key]))
            return out

    @staticmethod
    def get_raw_data_indoor(dir, filename, rep=-1):
        import glob
        poses = {}
        images = {}
        c = {}

        locations = sorted(glob.glob(os.path.join(dir, '*')))
        num_class = len(locations)

        nt = 0
        dirs = SortedDict()
        locs = {}

        for location in locations:
            if rep<0 or rep==nt:
                dirs[nt] = []
                with open(location + '/' + filename, 'r') as f:
                    for line in f:
                        if len(line)>10:
                            num = int(line[8:-2])
                        else:
                            num = int(line[8:])
                        str = 'seq-{0:02d}'.format(num)
                        dirs[nt].append(os.path.join(location, str))
            locs[nt] = location
            nt += 1

        for d in dirs:
            nt = 0

            for l in dirs[d]:
                imgs = glob.glob(os.path.join(l, '*color.png'))
                for ig in imgs:
                    cs = np.zeros((num_class))
                    cs[d] = 1
                    # c.append(cs)
                    c[ig] = cs

                    pose_file = ig[:-9] + 'pose.txt'
                    [mx, t] = Utils.load_poses(pose_file)

                    #img = cv2.imread(ig)
                    #if img is None:
                    #    print(ig)
                    #    raise Exception(ig)
                    # img = cv2.resize(img, (455, 256))
                    images[ig] = ig #.append(img)

                    q4 = Utils.mtx_2_q(mx)
                    poses[ig] = (t[0], t[1], t[2], q4[0], q4[1], q4[2], q4[3])
                    nt += 1

            print('{} {} {}'.format(locs[d], nt, d))

        #print("total images = {}".format(len(images)))
        return datasource(images, poses, c)

    @staticmethod
    def  mtx_2_q(mx):
        qw = np.sqrt(1 + mx[0][0] + mx[1][1] + mx[2][2]) / 2
        qx = (mx[2][1] - mx[1][2]) / (4 * qw)
        qy = (mx[0][2] - mx[2][0]) / (4 * qw)
        qz = (mx[1][0] - mx[0][1]) / (4 * qw)
        return qw, qx, qy, qz

    @staticmethod
    def load_poses(pose_file):
        mx = []
        t = []
        with open(pose_file, 'r') as fp:
            for line in fp.readlines():
                vals = map(float, line.split('\t')[:-1])
                mx.append(vals[:3])
                t.append(vals[3])
                if len(t)==3:
                    break
        mx = np.array(mx)
        return mx, t

    @staticmethod
    def get_raw_data(dir, filename, rep=-1):
        import glob
        poses = []
        images = []
        c = []

        locations = glob.glob(os.path.join(dir, '*'))
        num_class = len(locations)

        nt = 0
        for location in sorted(locations):
            if rep<0 or rep==nt:
                with open(location + '/' + filename, 'r') as f:
                    next(f)  # skip the 3 header lines
                    next(f)
                    next(f)
                    cnt = 0
                    for line in f:
                        fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                        p0 = float(p0)
                        p1 = float(p1)
                        p2 = float(p2)
                        p3 = float(p3)
                        p4 = float(p4)
                        p5 = float(p5)
                        p6 = float(p6)
                        img = cv2.imread(location + '/' + fname)
                        if img is None:
                            print(fname)
                            raise Exception(fname)
                        img = cv2.resize(img, (455, 256))
                        poses.append((p0, p1, p2, p3, p4, p5, p6))
                        images.append(img)
                        cs = np.zeros((num_class))
                        cs[nt] = 1
                        c.append(cs)
                        cnt += 1
                    print('{} {} {}'.format(location, nt, cnt))
            nt += 1

        print("total images = {}".format(len(images)))
        return datasource(images, poses, c)

    @staticmethod
    def get_data(rds, rep=-1, r0=0.2, ss=256):

        images = Utils.preprocess(rds.images, r0, ss=ss)
        #print 'Input image', len(images), len(rds.images), rep
        if rep < 0:
            return datasource(images, rds.poses, rds.xys)
        else:
            imgs = {}
            poses = {}
            c = {}
            #for a in range(len(images)):
            #print images.keys()
            for a in images:
                # print a, rep
                # print rds.xys[a], type(rep)
                # print  rds.xys[a][rep]
                if rds.xys[a][rep] == 1:
                    #print a
                    imgs[a] = images[a]
                    poses[a] = rds.poses[a]
                    c[a] = rds.xys[a]
            # print 'Output image', len(imgs)
            return datasource(imgs, poses, c)

    @staticmethod
    def gen_data(source):
        while True:
            keys = source.images.keys()
            indices = range(len(keys))
            random.shuffle(indices)
            for a in indices:
                i = keys[a]
                image = np.squeeze(source.images[i])
                pose_x = source.poses[i][0:3]
                pose_q = source.poses[i][3:7]
                xy = source.xys[i].reshape((1, 1, len(source.xys[i])))
                yield image, pose_x, pose_q, xy

    @staticmethod
    def gen_data_batch(source, batch_size):
        data_gen = Utils.gen_data(source)
        while True:
            image_batch = []
            pose_x_batch = []
            pose_q_batch = []
            xy_batch = []
            for _ in range(batch_size):
                image, pose_x, pose_q, xy = next(data_gen)
                image_batch.append(image)
                pose_x_batch.append(pose_x)
                pose_q_batch.append(pose_q)
                xy_batch.append(xy)
            xy_array = np.array(xy_batch).astype(float)
            # sz = xy_array.shape
            # xy_array = xy_array.reshape((sz[0], 1, 1, sz[1]))
            yield np.array(image_batch), \
                  np.array(pose_x_batch), \
                  np.array(pose_q_batch), \
                  xy_array

    @staticmethod
    def preprocess(images, r0=0.2, ss=256.0):
        images_out = {}
        images_cropped = {}
        for ig in images:

            if random.random() < r0:

                X = cv2.imread(ig)

                sz = X.shape[:2]
                r = min(sz[0]/ss, sz[1]/ss)
                sz = (int(sz[0]/r), int(sz[1]/r))
                X = cv2.resize(X, sz)
                X = Utils.centeredCrop(X, 224)
                # X, wh = Utils.randCrop(X, 224)
                images_cropped[ig] = X

        # compute images mean
        # N = 0
        # if not os.path.exists(Utils.MEAN_IMAGE):
        #    # mean = np.zeros((1, 3, 224, 224))
        #    mean = np.zeros((3, 224, 224))
        #    for X in images_cropped:
        #        # A = np.transpose(X, (2, 0, 1))
        #        X = np.asarray(X, np.float)
        #        mean[0] += X[:, :, 0]
        #        mean[1] += X[:, :, 1]
        #        mean[2] += X[:, :, 2]
        #        N += 1
        #    mean /= N
        #
        #    mean = mean.astype(np.uint8)
        #    mean = np.transpose(mean, (1, 2, 0))
        #    cv2.imwrite(Utils.MEAN_IMAGE, mean)

        # mean = cv2.imread(Utils.MEAN_IMAGE)
        # mean = np.transpose(mean, (2, 0, 1)).astype(np.float)

        # Subtract mean from all images
        for ig in images_cropped:
            # X = np.transpose(X, (2, 0, 1))
            X = np.asarray(images_cropped[ig], np.float)
            X[:, :, 0] -= 128  # mean[0]
            X[:, :, 1] -= 128  # mean[1]
            X[:, :, 2] -= 128  # mean[2]
            X = np.reshape(X, (1, 224, 224, 3))
            # X = np.transpose(X, (1, 2, 0))
            images_out[ig] = X

        # print len(images), len(images_out), len(images_cropped)
        return images_out

    @staticmethod
    def centeredCrop(img, output_side_length):
        height, width, depth = img.shape
        height_offset = int((height - output_side_length) / 2)
        width_offset = int((width - output_side_length) / 2)
        cropped_img = img[height_offset:height_offset + output_side_length,
                      width_offset:width_offset + output_side_length]
        return cropped_img

    @staticmethod
    def randCrop(img, output_side_length):
        height, width, depth = img.shape
        height_offset = np.random.randint(0, height - output_side_length)
        width_offset = np.random.randint(0, width - output_side_length)
        cropped_img = img[height_offset:height_offset + output_side_length,
                      width_offset:width_offset + output_side_length]

        height_offset -= (height - output_side_length) / 2
        width_offset -= (width - output_side_length) / 2
        return cropped_img, [height_offset, width_offset]

    @staticmethod
    def save_p_data(nnn, netFile):
        tmp_file = netFile + '.tmp'
        with open(tmp_file, 'w') as fp:
            pickle.dump(nnn, fp)
        shutil.copy(tmp_file, netFile)
        os.remove(tmp_file)

    @staticmethod
    def save_tf_data(saver, sess, netFile):
        net_dir = netFile[:-3]
        tmp_dir = net_dir + '_tmp'
        tmp_file = tmp_dir + '/fc'
        saver.save(sess, tmp_file)
        if os.path.exists(net_dir):
            shutil.rmtree(net_dir)
        shutil.move(tmp_dir, net_dir)

    @staticmethod
    def run_cmd(cmd):
        print(cmd)
        os.system(cmd)

    @staticmethod
    def get_A_T(Q):
        A = Utils.get_A(Q[:3, :3])
        # A = np.array(BlueNoteSensorRotation.
        #                 get_rotation_angles(Q[:3, :3], RotationSequence.XZY))
        T = Q[:3, 3]
        return A, T

    @staticmethod
    def get_relative_A_T(Q1, Q2):
        Q_inv = np.linalg.inv(Q1)
        Q = Q_inv.dot(Q2)
        raise Exception()
        return Utils.get_A_T(Q)

    @  staticmethod
    def create_Q(A, T):
        M = np.array(BlueNoteSensorRotation.rotation_matrix
                     (A[0], A[1], A[2], sequence=RotationSequence.XZY))
        M = np.concatenate((M, np.array(T).reshape(3, 1)), axis=1)
        M = np.concatenate((M, np.array([0,0,0,1]).reshape(1,4)))
        #M[:3, 3] = T
        raise Exception()
        return M

    @  staticmethod
    def create_M(A):
        A /= 180.0 / np.pi
        M = tr.euler_matrix(A[0], A[2], A[1])
        # M = np.array(BlueNoteSensorRotation.rotation_matrix
        #              (A[0], A[1], A[2], sequence=RotationSequence.XZY))
        return M[:3, :3]

    @staticmethod
    def get_A(Q):
        A = tr.euler_from_matrix(Q[:3, :3], axes='sxzy')
        A = np.array(A) * 180.0/np.pi
        #A = np.array(BlueNoteSensorRotation.
        #                 get_rotation_angles(Q[:3, :3], RotationSequence.XZY))
        return A

    @staticmethod
    def q_to_A(q):
        A = tr.euler_from_quaternion(q)
        A = np.array(A) * 180.0/np.pi
        return A

    @staticmethod
    def q_from_m(m):
        A = tr.quaternion_from_matrix(m)
        return A

    @staticmethod
    def q_to_m(q):
        A = tr.quaternion_matrix(q)
        return A

    @staticmethod
    def reshuffle_b(bucket):
        for id in bucket:
            for b in bucket[id]:
                np.random.shuffle(b[0])

if __name__ == "__main__":
    #pose = '-5.591326e-02 5.120575e-02 -9.971217e-01 -1.450234e+02 -2.512261e-02 9.982956e-01 5.267479e-02 -5.651594e+00 9.981195e-01 2.799552e-02 -5.453153e-02 3.692862e+02'
    #mat = np.array(list(map(float , pose.split(' ')))).reshape((3,4))

    mat =[[ 9.99990614e-01,  2.11742669e-03,  3.77992169e-03],
 [-2.11779305e-03,  9.99997753e-01,  9.29212715e-05],
 [-3.77971644e-03, -1.00925491e-04,  9.99992852e-01]]
    mat = np.array(mat)
    I = mat[:3, :3].dot(mat[:3, :3].transpose())
    A = Utils.get_A(mat)/180*np.pi
    print(A, np.linalg.norm(A))