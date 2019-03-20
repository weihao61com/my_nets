from LANL_Utils import l_utils, HOME
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

sys.path.append('{}/my_nets'.format(HOME))
from utils import Utils


def process(nx, ny, e):
    x = []
    y = []
    yp = []
    for a in range(len(nx)):
        if ny[a]>0.0:
            x.append(nx[a])
            y.append(ny[a])
        b = ny[a]
        if b<0.5:
            b += 1
        yp.append(b)
    z = np.polyfit(y, x, 2)
    x=np.array(x)
    y=np.array(y)
    error = []
    ny = []
    b = z[1] * z[1] - 4 * z[0] * (z[2] - x)
    c = z[0]*y*y +z[1]*y + z[2]

    plt.subplot(1, 2, 1)
    plt.plot(y, x, '.')
    plt.subplot(1, 2, 2)
    plt.plot(y, e, '.')
    plt.show()

    error.append(0)
    #
    # for a in range(len(x)):
    #     a1 = (np.sqrt(b[a])-z[1])/2/z[0]
    #     ny.append(a1)
    #     error.append(y[a] - a1)
    # plt.subplot(1,3,1)
    # plt.plot(y,x,'.')
    # plt.subplot(1,3,2)
    # plt.plot(y,ny,'.')
    # plt.subplot(1,3,3)
    # plt.hist(ny, 40)
    # plt.show()
    return np.average(np.abs(error)), z


def fft_refit(config):
    cfg = Utils.load_json_file(config)
    eval_file = cfg['eval_file'].format(HOME)
    rst_file = cfg['refit_file'].format(HOME)
    print eval_file

    dd = np.array(Utils.read_csv(eval_file)).astype(float)

    print process(dd[:, 3], dd[:, 1], dd[:, 5])
    print process(dd[:, 4], dd[:, 1], dd[:, 5])

    data = {}
    for d in dd:
        if not d[0] in data:
            data[d[0]] = []
        data[d[0]].append(d)

    e1 = []
    e2 = []
    refit = {}
    for c in data:
        dd = np.array(data[c])
        a1, z1 = process(dd[:, 3], dd[:, 1], dd[:, 5])
        a2, z2 = process(dd[:, 4], dd[:, 1], dd[:, 5])
        print dd.shape, a1, a2, z1, z2
        refit[c] = [z1,z2]
        e1.append(a1)
        e2.append(a2)
    print np.average(np.array(e1)), np.average(np.array(e2))

    with open(rst_file, 'w') as fp:
        pickle.dump(refit, fp)



if __name__ == '__main__':
    config = 'config.json'
    fft_refit(config)