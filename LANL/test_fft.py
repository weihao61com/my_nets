import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils
import glob
import cmath
from scipy import fftpack

seg = 150000


filename = '/home/weihao/Downloads/test'
files = glob.glob(os.path.join(filename, '*.csv'))
print 'total file', len(files)

vy = []
vx0 = []
vx = []
vs = []
for fnm in files:

    x = []
    with open(fnm, 'r') as fp:
        lines = fp.readlines()
    vy.append(os.path.basename(fnm)[:-4])
    lines = lines[1:]

    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])
    v = l_utils.get_core(x)
    print v


#print len(vx0), vx.shape
#print l_utils.csv_line(vy)
# print l_utils.csv_line(vx0)
#for a in range(vx.shape[1]):
#    print l_utils.csv_line(vx[:, a])
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