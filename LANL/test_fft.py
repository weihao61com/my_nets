import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils
import glob

seg = 150000


filename = '/home/weihao/Downloads/test'
files = glob.glob(os.path.join(filename, '*.csv'))
print 'total file', len(files)

vy = []
vx0 = []
vx = []
for f in files[:50]:

    x = []
    with open(f, 'r') as fp:
        lines = fp.readlines()
    vy.append(os.path.basename(f)[:-4])
    lines = lines[1:]

    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])

    x = np.array(x)
    f = np.fft.fft(x)
    f = abs(f)

    vx0.append(f[0])

    f = f[1:len(f)/2+1]
    af = abs(f).reshape(75, 1000)
    mf = np.mean(af, 1)
    vx.append(mf)

vx = np.array(vx)

print len(vx0), vx.shape
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