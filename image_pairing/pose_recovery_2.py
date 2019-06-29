import sys
import numpy as np
import evo.core.transformations as tr

HOME = '/home/weihao/Projects/'
if sys.platform == 'darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
from utils import Utils

if __name__ == '__main__':
    result_file = '/home/weihao/tmp/test.csv'
    data = Utils.read_csv(result_file)
    print len(data)
    output_file = '/home/weihao/tmp/rst.csv'
    rg = 3

    skip = (rg - 1) * rg * 2

    # A0 = [4.281100483954916	,66.64036592495356,10.843025665174292]
    # T0 = [-0.6582703,-0.66375178,0.51635194]
    A0 = [0, 0, 0]
    T0 = [0, 0, 0]

    Q = Utils.create_Q(A0, T0)
    print Q
    print A0, T0
    P = Utils.create_Q(A0, T0)  # np.identity(4)

    r = np.pi / 180
    fp = open(output_file, 'w')
    a = skip
    C1 = None
    Ds = []
    while a<len(data):
        A1s = []
        A2s = []

        for b in range(rg*2):
            d = np.array(map(float, data[a]))
            a += 1
            # X = Utils.create_Q(d[0:6:2] / 10, d[6:12:2] / 5)
            # Y = Utils.create_Q(d[1:6:2] / 10, d[7:12:2] / 5)
            X = Utils.create_M(d[0:6:2] / 10)
            Y = Utils.create_M(d[1:6:2] / 10)

            # Q = Q.dot(Utils.create_Q(d[0:6:2] / 10, d[6:12:2] / 5))
            # P = P.dot(Utils.create_Q(d[1:6:2] / 10, d[7:12:2] / 5))
            if b>=rg:
                X = np.linalg.inv(X)
                Y = np.linalg.inv(Y)
            A1 = Utils.get_A(X)
            A2 = Utils.get_A(Y)
            A1s.append(A1)
            A2s.append(A2)

            # for b in range(3):
            #     fp.write('{},{},'.format(A1[b], A2[b]))
            # for b in range(3):
            #     fp.write('{},{},'.format(T1[b], T2[b]))
            # fp.write('\n')
        #print 'A1', np.array(A1s)
        #print 'A2', np.array(A2s)
        #print
        B1 = A1s[2]
        B2 = A2s[2]
        if C1 is not None:
            D1 = (B1 + C1) / 2
            D2 = (B2 + C2) / 2
            Ds.append(np.linalg.norm(B2-B1))
        C1 = A1s[3]
        C2 = A2s[3]

    fp.close()
    Ds = np.array(Ds)
    print np.median(Ds), np.average(Ds*Ds)
