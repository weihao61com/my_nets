import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils
from scipy import fftpack, fft


def plot_line(lines):
    x = []
    y = []
    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])
    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.subplot(2, 1, 2)
    d = abs(fft(x))
    plt.plot(np.log(d[1:]))

    plt.show()

seg = 2048

filename = '/home/weihao/tmp/L/L_11.csv' #sys.argv[1]
#id = 10000000 # int(sys.argv[2])

a = 165000# int(sys.argv[1])
with open(filename, 'r') as fp:
    line0 = fp.readlines()

print 'total data', len(line0)
line0 = line0[a:a+10000]#l_utils.SEGMENT]

plot_line(line0)

step = int(len(line0)-seg)/100
m0 = None
dim = 256
sp = int(dim*0.4)
A1 = []
A2 = []
A3 = []
A4 = []
fp = open('/home/weihao/tmp/r.csv', 'w')
for id in range(0, len(line0)-seg, step):
    lines = line0[id:id+seg]

    x = []
    y = []
    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])

    v = l_utils.feature_final(x, False, dim)
    y = np.mean(np.array(y))
    if m0 is None:
        m0 = y
    #plt.plot(v[1:])
    #plt.show()
    fp.write('{},{},{},{},{},{}\n'.
             format(id, m0-y, np.mean(v[1:sp]), np.std(v[1:sp]),np.mean(v[sp:]), np.std(v[sp:])))
    A1.append(np.mean(v[1:sp]))
    A2.append(np.std(v[1:sp]))
    A3.append(np.mean(v[sp:]))
    A4.append(np.std(v[sp:]))

fp.close()
z1 = np.polyfit(A1, A2, 2)
z2 = np.polyfit(A4, A3, 1)

print m0
print z1
print z2
plt.plot(v[1:])
plt.show()