import os
import glob
from scipy import fftpack, fft
import numpy as np
import matplotlib.pyplot as plt
import sys

SEG = 150000

location = '/home/weihao/Downloads/test' #sys.argv[1]
files = glob.glob(os.path.join(location, 'seg_*.csv'))

fig, ax = plt.subplots()
dct = True
nt = 0
NT = 10
st = int(sys.argv[1])

for f in files[st:NT+st]:
    with open(f, 'r') as fp:
        lines = fp.readlines()
        x = []
        for line in lines[1:]:
            x.append(float(line[:-1]))
        if dct:
            d = fftpack.dct(x)
            dv = np.array(d[1:100001]).reshape(1000,100)
            dv = np.std(dv, 1)
            ax.plot(dv[400:], c='{}'.format(float(nt)/NT), label='{}'.format(f))
            print '{0} {1:5.2f} {2:5.2f}'.format(f, d[0]/len(x), dv[180])
        else:

            d = abs(fft(x))
            dv = np.array(d[1:50001]).reshape(1000, 50)
            dv = np.mean(dv, 1)
            ax.plot(dv[:400], c='{}'.format(float(nt)/NT), label='{}'.format(f))
            print '{0} {1:5.2f} {2:9.0f}'.format(f, d[0].real / len(x), dv[180])
    nt += 1
leg = ax.legend()
ax.legend(loc='upper left', frameon=False)

plt.show()