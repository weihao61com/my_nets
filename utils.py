import cv2
import numpy as np
import os
import random
import json
from sortedcontainers import SortedDict


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

    @staticmethod
    def calculate_loss(v1, v2):
        diff = v1- v2
        r = np.linalg.norm(diff, axis=1)
        loss = np.linalg.norm(r)
        return loss*loss/len(r), np.median(r)

    @staticmethod
    def load_json_file(filename):
        out = SortedDict()
        with open(filename, 'r') as fp:
            js = json.load(fp)
            for key in js:
                if not key.startswith('_'):
                    out[key] = js[key]
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
