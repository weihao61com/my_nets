import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils
import glob
import cmath

seg = 150000


filename = '/home/weihao/Downloads/test'
files = glob.glob(os.path.join(filename, '*.csv'))
print 'total file', len(files)

vy = []
vx0 = []
vx = []
vs = []
for fnm in files[:10]:

    x = []
    with open(fnm, 'r') as fp:
        lines = fp.readlines()
    vy.append(os.path.basename(fnm)[:-4])
    lines = lines[1:]

    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])

    #v = 0
    avg = l_utils.fft_feature_final(x, 100, 10000, True)
    #std = [0]
    #v, avg ,std = l_utils.fft_features(x, 100)

    # vx0.append(v)
    vx.append(avg)
    #vs.append(std)

    f = np.fft.rfft(x)
    # plt.subplot(2, 1, 1)
    n1 = []
    n2 = []
    for ff in f:
        n1.append(abs(ff))
        n2.append(cmath.phase(ff))
    plt.subplot(2, 1, 1)
    plt.plot(avg[1:])
    plt.subplot(2, 1, 2)
    plt.plot(n1[100:])
    # plt.plot(avg)
    # plt.subplot(2, 2, 4)
    # plt.plot(std)
    plt.show()
    print ''

vx = np.array(vx)
vs = np.array(vs)

#print len(vx0), vx.shape
print l_utils.csv_line(vy)
# print l_utils.csv_line(vx0)
for a in range(vx.shape[1]):
    print l_utils.csv_line(vx[:, a])
#for a in range(vs.shape[1]):
#    print l_utils.csv_line(vs[:, a])


#
# plt.subplot(3, 1, 1)
# plt.plot(abs(f))
# plt.subplot(3, 1, 2)
# plt.plot(np.log(abs(f.real)))
# plt.subplot(3, 1, 3)
# plt.plot(np.log(abs(f.imag)))
# plt.show()