from LANL_Utils import l_utils
import sys
import numpy as np


def process(x, y):
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    error = []
    for a in range(len(x)):
        error.append(y[a] - p(x[a]))

    print np.average(np.abs(error))


HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))

from utils import Utils

eval_file = '/home/weihao/tmp/fit.csv'

dd = np.array(Utils.read_csv(eval_file)).astype(float)

process(dd[:, 3], dd[:, 1])
process(dd[:, 4], dd[:, 1])

data = {}
for d in dd:
    if not d[0] in data:
        data[d[0]] = []
    data[d[0]].append(d)

for c in data:
    dd = np.array(data[c])
    print dd.shape
    process(dd[:, 3], dd[:, 1])
    process(dd[:, 4], dd[:, 1])

