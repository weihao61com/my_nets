import sys
import numpy as np
import evo.core.transformations as tr


HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
from utils import Utils


if __name__ == '__main__':
    result_file = '/home/weihao/tmp/test.csv'
    data = Utils.read_csv(result_file)
    print len(data)
    output_file = '/home/weihao/tmp/rst.csv'
    rg = 3

    skip = (rg-1)*rg

    #A0 = [4.281100483954916	,66.64036592495356,10.843025665174292]
    #T0 = [-0.6582703,-0.66375178,0.51635194]
    A0 = [0,0,0]
    T0 = [0,0,0]

    Q = Utils.create_Q(A0, T0)
    print Q
    print A0, T0
    P = Utils.create_Q(A0, T0) # np.identity(4)

    r = np.pi/180
    fp = open(output_file, 'w')
    for a in range(len(data)):
        d = np.array(map(float, data[a]))
        Q = Q.dot(Utils.create_Q(d[0:6:2]/10, d[6:12:2]/10))
        P = P.dot(Utils.create_Q(d[1:6:2]/10, d[7:12:2]/10))
        A1, T1 = Utils.get_A_T(Q)
        A2, T2 = Utils.get_A_T(P)
        for b in range(3):
            fp.write('{},{},'.format(A1[b], A2[b]))
        for b in range(3):
            fp.write('{},{},'.format(T1[b], T2[b]))
        fp.write('\n')
        #if a==998:
        #    break

    fp.close()