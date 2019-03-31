import sys
import pickle
from LANL_Utils import l_utils, HOME
#import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedDict

sys.path.append('{}/my_nets'.format(HOME))
from utils import Utils

#
# fnm = '/home/weihao/Projects/p_files/L_0.p'
# with open(fnm, 'r') as fp:
#     d = pickle.load(fp)
#
# nt = 0
# for a in d:
#     if nt%100==0:
#         str = l_utils.csv_line(a[1])
#         print '{},{}'.format(a[0], str)
#     nt += 1


# filename = '/home/weihao/Projects/tmp/rst.csv'
# x = []
# y = []
# with open(filename, 'r') as fp:
#     for line in fp.readlines()[1:]:
#         str = line.split(',')
#         x.append(float(str[1]))
#         #y.append(float(str[3]))
#
#
# n, bins, patches = plt.hist(x, 100)
# #print n, bins
# plt.show()

count = {}
filename = '/Users/weihao/Projects/tmp/count.txt'
dd = Utils.read_csv(filename, '\t')
for d in dd:
    if len(d) > 4:
        if not d[2] in count:
            count[d[2]] = float(d[6])

fc = SortedDict()
for f in count:
    fc[count[f]] = f

filename = '/Users/weihao/Projects/tmp/fit.csv'
dd = np.array(Utils.read_csv(filename))
data = SortedDict()
for d in dd:
    if not d[1] in data:
        data[d[1]] = []
    data[d[1]].append(d[2:])

a = 1

for c in fc:
    f = fc[c]
    if f not in data:
        continue
    dd = np.array(data[f]).astype(float)
    print f, count[f]
    plt.subplot(4, 8, a)
    a += 1
    plt.plot(dd[:, 0], dd[:, 2], '.r', dd[:, 0], dd[:, 3], '.b')
    plt.subplot(4, 8, a)
    a += 1
    plt.plot(dd[:, 0], dd[:, 4])
plt.show()