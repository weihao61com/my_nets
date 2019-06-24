import numpy as np
from sortedcontainers import SortedDict
from bluenotelib.common.coordinate_transforms \
    import Quaternion, BlueNoteSensorRotation, RotationSequence
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def linear_intepolation(k1, k2, v1, v2, k):
    dk = k2-k1
    dv = v2-v1
    v = v1 + (k-k1)*dv/dk
    return v


def get_intepolation(data, keys):
    a = 1
    b = 0
    dk = data.keys()
    idx = SortedDict()

    while a<len(data) and b<len(keys):
        if dk[a] < keys[b]:
            a += 1
        else:
            idx[b] = a
            b += 1

    if b<len(keys):
        b += 1
        idx[b] = a

    out = SortedDict()
    for b in idx:
        a = idx[b]
        k1 = dk[a]
        k2 = dk[a-1]
        v1 = data[k1]
        v2 = data[k2]
        k = keys[b]
        out[k] = linear_intepolation(k1, k2, v1, v2, k)

    return out


def load_kitti(locs, seq):
    poses_file = '{}/poses/{}.txt'.format(locs, seq)
    poses = []
    with open(poses_file, 'r') as fp:
        for line in fp.readlines():
            d = map(float, line.split(' '))
            d = np.array(d).reshape((3,4))
            A = BlueNoteSensorRotation.get_rotation_angles(d[:, :3], RotationSequence.XZY)
            poses.append(np.array([np.array(A), d[:, 3]]).reshape(6))

    time_file = '{}/sequences/{}/times.txt'.format(locs, seq)
    out = SortedDict()
    nt = 0
    with open(time_file, 'r') as fp:
        for line in fp.readlines():
            t = float(line)
            out[t] = poses[nt]
            nt += 1

    if nt!=len(poses):
        raise Exception("Number not match")

    return out


def load_rst(rst_location):
    out = SortedDict()
    with open(rst_location, 'r') as fp:
        for line in fp.readlines():
            d = map(float, line.split(' '))
            q = Quaternion(qw=d[7], qx=d[4], qy=d[5], qz=d[6])
            r = Quaternion.to_rotation_matrix(q)
            A = BlueNoteSensorRotation.get_rotation_angles(r, RotationSequence.XZY)
            out[d[0]] = np.array(list(A)+list(d[1:4]))

    return out

if __name__=="__main__":
    data_location = '/home/weihao/Projects/datasets/kitti'
    seq ='02'
    rst_location = '/home/weihao/GITHUB/ORB_SLAM2-master/Examples/Monocular/KeyFrameTrajectory.txt'
    output = '/home/weihao/tmp/orb.csv'

    data = load_kitti(data_location, seq)
    rst = load_rst(rst_location)

    data = get_intepolation(data, rst.keys())

    X = []
    Y = []
    for key in data:
        d = data[key]
        r = rst[key]
        Y.append(d[-3:])
        X.append(r[-3:])


    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    print regr.coef_
    print regr.intercept_

    err = []

    with open(output, 'w') as fp:
        for key in data:
            d = data[key]
            r = rst[key]
            pre = regr.predict([r[-3:]])[0]
            err.append(np.linalg.norm(pre-d[-3:]))
            fp.write('{}'.format(key))
            for a in range(len(d)):
                if a<3:
                    fp.write(',{},{}'.format(d[a], r[a]))
                else:
                    fp.write(',{},{},{}'.format(d[a], pre[a-3], r[a]))
            fp.write('\n')
    err=np.array(err)
    print np.sqrt(np.mean(err*err))