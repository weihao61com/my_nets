# from scipy import stats
# from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
import numpy as np
import sys

filename = '/home/weihao/tmp/r.txt'
if sys.platform=='darwin':
    filename = '/Users/weihao/tmp/r.txt'

def p_fit(x, y):
    v = np.polyfit(x, y, 2)
    return v

#w, h = figaspect(0.5)
#fig = plt.figure(figsize=(w, h))
usecols = (2,7)
if len(sys.argv)>1:
    usecols = map(int, sys.argv[1].split(','))

if len(sys.argv)>2:
    filename = sys.argv[2]

da = np.loadtxt(filename, usecols=usecols)

mx = max(da[:, 0])

length = len(da)
l2 =int(length/2)

l3 = length-min(l2, 200)

#for a in range(int(l2), 0, -1):

for a in range(int(l2)):
    v = p_fit(da[(l3-a):length-a, 0], da[(l3-a):length-a, 1])
    pre1 = v[0] * da[(l3-a):length-a, 0] *da[(l3-a):length-a, 0] + v[1] * da[(l3-a):length-a, 0] + v[2]
    mx = max(da[(l3-a):length-a, 0])
    print '{2} {0:.9f} {1:.9f}'.format(pre1[-1], 2 * mx * v[0] + v[1], a)
    plt.plot(da[(l3-a):length-a, 0], pre1, linewidth=2)

#print slope1, intercept1
#print pre1[-1], pre2[-1]
#plt.show()
