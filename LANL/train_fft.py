import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils
import glob

SEG = 150000


def get_features(lines):
    if not len(lines)==SEG:
        raise Exception("Wrong data length {}".format(len(lines)))

    x = []
    y = []
    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])

    v0, avg, std = l_utils.fft_features(x, 250)
    y = np.array(y)
    mn = np.mean(y)
    print mn, np.std(y)
    return mn,v0, avg, std


fid = 10
p = 900000

location = '/home/weihao/tmp' #sys.argv[1]
files = glob.glob(os.path.join(location, 'L_*.csv'))
vx = []
vx0 = []
vy = []

filename = files[fid]
print filename
with open(filename, 'r') as fp:
    lines = fp.readlines()

A = get_features(lines[p:p+SEG])

vx.append(A[2])
vx0.append(A[1])
vy.append('S{}'.format(int(A[0]*1000)))

vx = np.array(vx)

print l_utils.csv_line(vy)
print l_utils.csv_line(vx0)
for a in range(vx.shape[1]):
    print l_utils.csv_line(vx[:, a])



#
# plt.subplot(3, 1, 1)
# plt.plot(abs(f))
# plt.subplot(3, 1, 2)
# plt.plot(np.log(abs(f.real)))
# plt.subplot(3, 1, 3)
# plt.plot(np.log(abs(f.imag)))
# plt.show()