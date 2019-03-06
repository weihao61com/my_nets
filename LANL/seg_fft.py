import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils

seg = 150000

def csv_line(dd):
    output = None
    for d in dd:
        if output is None:
            output = '{}'.format(d)
        else:
            output = '{},{}'.format(output, d)

    return output

filename = sys.argv[1]
# id = int(sys.argv[2])
max_seg = 130

with open(filename, 'r') as fp:
    lines = fp.readlines()

nseg = len(lines)/seg
print 'total data', len(lines)

if len(lines)>max_seg*seg:
    rm = -seg*max_seg
else:
    rm = len(lines)%seg
lines = lines[rm:]
nseg = len(lines)/seg
print 'total data', len(lines), nseg

vy = []
vx = []
vx0 = []
for id in range(nseg-1, 0, -10):
    ls = lines[id*seg:(id+1)*seg]

    x = []
    y = []
    for line in ls:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])

    y = np.mean(np.array(y))
    vy.append('S{}'.format(int(y*1000)))

    x = np.array(x)
    f = np.fft.fft(x)
    f = abs(f)

    vx0.append(f[0])

    f = f[1:len(f)/2+1]
    af = abs(f).reshape(75, 1000)
    mf = np.std(af, 1)
    vx.append(mf)

vx = np.array(vx)

print len(vy), len(vx0), vx.shape
print csv_line(vy)
print csv_line(vx0)
for a in range(vx.shape[1]):
    print csv_line(vx[:, a])



#
# plt.subplot(3, 1, 1)
# plt.plot(abs(f))
# plt.subplot(3, 1, 2)
# plt.plot(np.log(abs(f.real)))
# plt.subplot(3, 1, 3)
# plt.plot(np.log(abs(f.imag)))
# plt.show()