from scipy import stats
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
import numpy as np
import sys

filename = '/home/weihao/tmp/r.txt'
w, h = figaspect(0.5)
fig = plt.figure(figsize=(w, h))
usecols = (1,2,3)
if len(sys.argv)>1:
    usecols = map(int, sys.argv[1].split(','))

da = np.loadtxt(filename, usecols=usecols)
print 'max', max(da[:, 0])

da[:, 0] = da[:, 0] - max(da[:, 0])
#print da
slope, intercept, r_value, p_value, std_err = stats.linregress(da[:, 0], da[:, 1])
pre1 = da[:, 0] *slope + intercept

ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(da[:, 0], da[:, 1],  color='black')
ax1.plot(da[:, 0], pre1, color='blue', linewidth=3)

slope1, intercept1, r_value, p_value, std_err = stats.linregress(da[:, 0], da[:, 2])
pre2 = da[:, 0] *slope1 + intercept1

ax1 = fig.add_subplot(1, 2, 2)
ax1.scatter(da[:, 0], da[:, 2],  color='black')
ax1.plot(da[:, 0], pre2, color='blue', linewidth=3)

print slope, intercept
print slope1, intercept1
plt.show()