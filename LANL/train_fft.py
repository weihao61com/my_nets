import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils
import glob
import random

SEG = 150000


def rdm_ids(files, cv):
    l = len(files)
    ids = range(l)
    random.shuffle(ids)
    dv = cv/float(l)
    idx = {}
    nx = 0
    for id in ids:
        idx[files[id]] = int(nx*dv)
        nx += 1
    return idx


def get_features(lines):
    if not len(lines)==SEG:
        raise Exception("Wrong data length {}".format(len(lines)))

    x = []
    y = []
    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])

    return np.mean(y), l_utils.fft_feature_final(x)

CV = 5
NF = 5000

out_loc = '/home/weihao/Projects/p_files'
location = '/home/weihao/tmp/L' #sys.argv[1]
files = glob.glob(os.path.join(location, 'L_*.csv'))
ids = rdm_ids(files, CV)

for c in range(CV):
    data = []
    for filename in files:
        if ids[filename] == c:
            with open(filename, 'r') as fp:
                lines = fp.readlines()
            print 'records', c, filename, len(lines),  len(lines)/SEG
            step = (len(lines) - SEG - 1)/NF
            for a in range(NF):
                p = a*step
                A= get_features(lines[p:p+SEG])
                data.append(A)
    print "Total data", c, len(data)
    filename = os.path.join(out_loc, 'L_{}.p'.format(c))
    with open(filename, 'w') as fp:
        pickle.dump(data, fp)

#
#
# vx = np.array(vx)
#
# print l_utils.csv_line(vy)
# for a in range(vx.shape[1]):
#     print l_utils.csv_line(vx[:, a])
#
# plt.subplot(3, 1, 1)
# plt.plot(abs(f))
# plt.subplot(3, 1, 2)
# plt.plot(np.log(abs(f.real)))
# plt.subplot(3, 1, 3)
# plt.plot(np.log(abs(f.imag)))
# plt.show()