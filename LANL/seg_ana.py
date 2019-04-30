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

def cal_feature(x):
    win = 4096
    sz = 16
    d = abs(fft(x[:win]))
    d = d.reshape((win/sz, sz))
    d = np.mean(d, 1)
    return d[1:win/sz/2+1]
    #sz = len(x)/win
    #b = x.reshape((win, sz))
    #d = abs(fft(np.mean(b, 0)))
    #return d[1:sz/2+1]

# LF
def cal_feature2(x):
    win = 300
    sz = len(x)/win
    b = x.reshape((win, sz))
    dd = []
    for a in range(win):
        d = abs(fft(b[a, :]))
        dd.append(d)

    dd = np.array(dd)
    d0 = np.mean(dd, 0)
    #d0 = np.std(dd, 0)
    #d0 = np.median(dd, 0)
    return d0[1:sz/2+1]
    #return d0[1:100]


# HF
def cal_feature1(x):
    win = 500
    sz = len(x)/win
    b = x.reshape((win, sz))
    dd = []
    for a in range(sz):
        d = abs(fft(b[:, a]))
        dd.append(d)

    dd = np.array(dd)
    d0 = np.mean(dd, 0)
    #d0 = np.std(dd, 0)
    #d0 = np.median(dd, 0)
    return d0[1:win/2+1]
    #return d0[1:100]

avg_spec = None
for f in range(1, 16):
    filename = '/Users/weihao/tmp/L/L_{}.csv'.format(f) #sys.argv[1]
    #id = 10000000 # int(sys.argv[2])

    with open(filename, 'r') as fp:
        lines = fp.readlines()

    v = map(float, lines[0].split(','))
    t0 = v[1]

    #print 'total data', len(lines)

    a1 = 0 #10000000 #int(sys.argv[1])
    line0 = lines[a1:a1+l_utils.SEGMENT]
    x, y = l_utils.get_values(line0)
    d1 = cal_feature(x)
    # d1 = cal_feature2(x)
    # for a in range(251):
    #     a1 = d1[0][a]
    #     a2 = d1[1][a]
    #     print a, a1.real, a1.imag, a2.real, a2.imag, abs(a1), abs(a2)
    #
    # raise Exception()
    #

    if avg_spec is None:
        avg_spec = d1.copy()
    #d1 -= avg_spec
    print np.mean(y)/t0, np.mean(d1), np.std(d1)

    a1 += (len(lines)-l_utils.SEGMENT)/3
    line0 = lines[a1:a1+l_utils.SEGMENT]
    x, y = l_utils.get_values(line0)
    d2 = cal_feature(x)
    #d2 -= avg_spec
    print np.mean(y)/t0, np.mean(d2), np.std(d2)

    a1 += (len(lines)-l_utils.SEGMENT)/3
    line0 = lines[a1:a1+l_utils.SEGMENT]
    x, y = l_utils.get_values(line0)
    d3 = cal_feature(x)
    #d3 -= avg_spec
    print np.mean(y)/t0, np.mean(d3), np.std(d3)

    a1 += (len(lines)-l_utils.SEGMENT)/3
    line0 = lines[a1:a1+l_utils.SEGMENT]
    x, y = l_utils.get_values(line0)
    d4 = cal_feature(x)
    #d4 -= avg_spec
    print np.mean(y)/t0, np.mean(d4), np.std(d4)

    #
    plt.subplot(2,2,1)
    plt.plot(d1)
    plt.subplot(2,2,2)
    plt.plot(d2)
    plt.subplot(2,2,3)
    plt.plot(d3)
    plt.subplot(2,2,4)
    plt.plot(d4)
    plt.plot(d3)
    plt.plot(d2)
    plt.plot(d1)
    plt.show()

raise Exception()

#    break

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
        v = map(float, line.split(','))
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