import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils


def get_fe(d):
    a=0
    b=1
    output = []
    while a<len(d):
        c = a+b
        mn = np.mean(d[a:c])
        output.append(mn)
        a += c
        b *= 2
    return output


# filename = '/home/weihao/Downloads/test/seg_0cca14.csv'
# filename = '/home/weihao/Downloads/train.csv'
filename = sys.argv[1] #'/home/weihao/tmp/L_1.csv'
max_line = 1e8
lines = []
header = None
with open(filename, 'r') as fp:
    while True:
        try:
            line = fp.next()
            if header is None:
                header = line[:-1]
                print header
            else:
                lines.append(line)
                if len(lines)>=max_line:
                    break
        except:
            break

nl = len(lines)
seg = 150000
rm = nl%seg
if rm>0:
    lines = lines[rm:]

nseg = len(lines)/seg
limit = 200
if nseg>limit:
    lines = lines[-limit*seg:]

print 'Seg', nseg, len(lines), 'removed', rm
nt = 0
data = []
t0 = datetime.datetime.now()
win = 40
nseg = len(lines)/seg
step = int(nseg/win)

pt = False
for a in range(nseg, 1, -step):

    end = a*seg
    start = end - seg

    data = []
    t = []
    for line in lines[start:end]:
        v = map(float, line.split(','))
        data.append(v[0])
        t.append(v[1])

    f = abs(np.fft.rfft(data))
    if pt:
        plt.subplot(1, 2, 1)
        plt.plot(data)
        plt.subplot(1, 2, 2)
        plt.plot(abs(f[1:]))

    f = l_utils.get_core(data)
    print a, nt, np.mean(t), f
    nt += 1

    if pt:
        plt.show()
