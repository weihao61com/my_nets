from scipy import stats
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
import numpy as np
import sys


def get_numbers(str):
    strs = str.split(',')
    output = []
    for s in strs:
        if '-' in s:
            num = s.split('-')
            n1 = int(num[0])
            n2 = int(num[1])
            for a in range(n1, n2+1):
                output.append(a)
        else:
            output.append(int(s))
    return output


filename = '/home/weihao/tmp/r.txt'
if sys.platform=='darwin':
    filename = '/Users/weihao/tmp/r.txt'

w, h = figaspect(0.5)
fig = plt.figure(figsize=(w, h))
usecols = (2,3,4,5,6)
if len(sys.argv)>1:
    usecols = get_numbers(sys.argv[1])

if len(sys.argv)>2:
    filename = sys.argv[2]

section = 0.5
if len(sys.argv)>3:
    section = float(sys.argv[3])

da = np.loadtxt(filename, usecols=usecols)

print 'max', max(da[:, 0])

zone = len(usecols) - 1

r = int(np.ceil(np.sqrt(zone)))
c = int(np.ceil(zone/float(r)))

#da[:, 0] = da[:, 0] - max(da[:, 0])
mx = max(da[:, 0])

#print da
for a in range(zone):
    length = int(len(da)*section)
    slope, intercept, r_value, p_value, std_err = stats.linregress(da[length:, 0], da[length:, a+1])
    pre1 = da[:, 0] *slope + intercept

    ax1 = fig.add_subplot(c, r, a+1)
    ax1.scatter(da[:, 0], da[:, a+1],  color='black', s=5)
    ax1.plot(da[:, 0], pre1, color='blue', linewidth=2)
    ymin = np.min(da[:, a+1])
    ymax = np.max(da[:, a+1])
    if ymax>ymin:
        if slope<0:
            ax1.set_ylim([ymin, min(ymax,intercept-slope*100)])
        else:
            ax1.set_ylim([ymin, ymax])
    #
    # slope1, intercept1, r_value, p_value, std_err = stats.linregress(da[:, 0], da[:, 2])
    # pre2 = da[:, 0] *slope1 + intercept1
    #
    # ax1 = fig.add_subplot(1, 2, 2)
    # ax1.scatter(da[:, 0], da[:, 2],  color='black')
    # ax1.plot(da[:, 0], pre2, color='blue', linewidth=3)

    print '{0:.9f} {1:.9f} {2:.9f}'.format(pre1[-1], intercept, slope)
    # print slope, intercept, pre1[-1]
#print slope1, intercept1
#print pre1[-1], pre2[-1]
plt.show()
