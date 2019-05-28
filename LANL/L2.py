import os
import glob
from scipy import fftpack, fft
import numpy as np
import matplotlib.pyplot as plt
import pywt


def W(x):
    cs = pywt.dwt(x, 'bior1.3')

    return None


def L2(x):
    sts = []
    while len(x) > 2:
        ll = len(x)
        x1 = x[0:ll:2]
        x2 = x[1:ll:2]
        a = x1 - x2
        x = (x1 + x2)
        sts.append(np.std(a))

    sts.append(np.mean(x))
    sts.append(np.std(x))
    return sts

id = 0 #int(sys.argv[1])
SEG = 4096

location = '/Users/weihao/tmp/L' #sys.argv[1]
files = glob.glob(os.path.join(location, 'L_*.csv'))
print len(files), files[id]
with open(files[id], 'r') as fp:
    lines = fp.readlines()

NF = 100
print 'total', len(lines)/SEG
step = (len(lines) - SEG - 1) / NF
fig, ax = plt.subplots()
dct = True

for a in range(NF):
    p = a * step
    x = []
    y = []
    for line in lines[p:p+SEG]:
        v = map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])

    x = np.array(x)
    y = np.array(y)

    sts = W(x)
    print np.mean(y),

    for s in sts:
        print s,
    print

