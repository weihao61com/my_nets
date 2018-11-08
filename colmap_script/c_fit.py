from scipy import stats
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
import numpy as np
import sys

filename = '/home/weihao/tmp/r.txt'
if sys.platform=='darwin':
    filename = '/Users/weihao/tmp/r.txt'

def p_fit(x, y):
    v = np.polyfit(x, y, 2)
    return v

w, h = figaspect(0.5)
fig = plt.figure(figsize=(w, h))
usecols = (1,9,10,11,12,13,14)
if len(sys.argv)>1:
    usecols = map(int, sys.argv[1].split(','))

if len(sys.argv)>2:
    filename = sys.argv[2]

da = np.loadtxt(filename, usecols=usecols)

print 'max', max(da[:, 0])

zone = (len(usecols) - 1)/2

r = int(np.ceil(np.sqrt(zone)))
c = int(np.ceil(zone/float(r)))

#da[:, 0] = da[:, 0] - max(da[:, 0])
mx = max(da[:, 0])

#print da
for a in range(zone):
    length = int(len(da)/2)
    data = da[:, a*2+1] - da[:, a*2+2]
    slope, intercept, r_value, p_value, std_err = stats.linregress(da[length:, 0], data[length:])
    pre1 = da[:, 0] *slope + intercept

    ax1 = fig.add_subplot(c, r, a+1)
    ax1.scatter(da[:, 0], data,  color='black', s=5)
    ax1.plot(da[:, 0], pre1, color='blue', linewidth=2)

    print slope, intercept, pre1[-1]
#print slope1, intercept1
#print pre1[-1], pre2[-1]
plt.show()
