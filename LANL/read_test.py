import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os

#filename = '/home/weihao/Downloads/test/seg_0cca14.csv'
filename = '/home/weihao/Downloads/train.csv'

skip = 1

start = 5620000*skip
stop  = 5680000*skip

nt = 0
data = []
header = None
t0 = datetime.datetime.now()
avg = 0
cnt = 0
m0 = 0

with open(filename, 'r') as fp:
    while True:
        try:
            line = fp.next()
            if header is None:
                header = line[:-1]
                print header
            else:
                v = map(float, line.split(','))
                nt += 1
                if stop>=nt>start:
                    if nt%skip==0:
                    #if abs(v[0])<100:
                        data.append(v)
                        avg += abs(v[-1]-m0)
                        cnt += 1
                if nt>=stop:
                    break
        except:
            break

print 'error', avg/cnt, avg/cnt*2624

data = np.array(data)
sz = data.shape

for a in range(sz[1]):
    plt.subplot(sz[1], 2, a*2+1)
    plt.plot(data[:, a])
    plt.subplot(sz[1], 2, a*2+2)
    plt.plot(data[1:, a]-data[:-1, a])

plt.show()
