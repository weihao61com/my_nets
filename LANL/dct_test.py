import os
import glob
from scipy import fftpack, fft
import numpy as np
import matplotlib.pyplot as plt
import sys

id = int(sys.argv[1])
SEG = 150000

location = '/home/weihao/tmp/L' #sys.argv[1]
files = glob.glob(os.path.join(location, 'L_*.csv'))
print files[id]
with open(files[id], 'r') as fp:
    lines = fp.readlines()

NF = 9
print 'total', len(lines)/SEG
step = (len(lines) - SEG - 1) / NF
fig, ax = plt.subplots()
dct = True

for a in range(NF):
    p = a * step
    x = []
    y = []
    for line in lines[p:p+SEG]:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])

    if dct:
        d = fftpack.dct(x)
        dv = np.array(d[1:100001]).reshape(1000,100)
        dv = np.std(dv, 1)
        ax.plot(dv[400:], c='{}'.format(float(a)/NF), label='{0:5.2f}'.format(np.mean(y)))
        print '{0:5.2f} {1:5.2f} {2:5.2f} {3}'.\
            format(np.mean(y), d[0]/len(x), dv[180], np.mean(dv[:400]))
    else:

        d = abs(fft(x))
        dv = np.array(d[1:50001]).reshape(1000, 50)
        #dv = np.std(dv, 1)
        dv = np.mean(dv, 1)
        ax.plot(dv[:400], c='{}'.format(float(a)/NF), label='{0:5.2f}'.format(np.mean(y)))
        print '{0:5.2f} {1:5.2f} {2:9.0f} {3}'.\
            format(np.mean(y), d[0].real / len(x), dv[180], np.mean(dv[:400]))
leg = ax.legend();
ax.legend(loc='upper left', frameon=False)

plt.show()