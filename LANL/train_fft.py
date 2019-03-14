import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils
import glob
import random
from sortedcontainers import SortedDict
from multiprocessing.dummy import Pool as ThreadPool

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils

SEG = 10000
CV = 5
dct = False
dim = 200
threads = 2
location = '/home/weihao/tmp/L'
out_location = '/home/weihao/Projects/p_files/L/L_{}'


def process(c):
    data = []
    for filename in files:
        if ids[filename] == c:
            with open(filename, 'r') as fp:
                lines = fp.readlines()
            NF = int (len(lines) / 10000)
            print 'records', c, filename, len(lines), len(lines) / SEG, NF
            rps = np.random.randint(0, len(lines) - SEG - 1, NF)
            rps.sort()
            pr = 0
            for a in range(NF):
                #rm = rps[a] - pr
                #lines = lines[rm:]
                A = l_utils.get_features(lines[rps[a]:rps[a]+SEG], dct, dim)
                data.append(A)
                #pr = rps[a]
    print "Total data", c, len(data)
    filename = os.path.join(out_loc, 'L_{}.p'.format(c))
    with open(filename, 'w') as fp:
        pickle.dump(data, fp)


if __name__ == '__main__':

    cfg = Utils.load_json_file('config.json')

for id in range(1000):
    out_loc = out_location.format(id)
    if not os.path.exists(out_loc):
        os.mkdir(out_loc)

        files = glob.glob(os.path.join(location, 'L_*.csv'))
        ids = l_utils.rdm_ids(files)

        numbers = [0,1,2,3,4]

        if threads>1:
            pool = ThreadPool(threads)
            results = pool.map(process, numbers)
            pool.close()
            pool.join()
        else:
            for c in numbers:
                process(c)

    if os.path.exists('STOP'):
        break
# process(c, dct, dim)


#
#
# vx = np.array(vx)
#
# print l_utils.csv_line(vy)
# for a in range(vx.shape[1]):
#     print l_utils.csv_line(vx[:, a])
#
# plt.subplot(3, 1, 1)
# plt.plot(abs(f))
# plt.subplot(3, 1, 2)
# plt.plot(np.log(abs(f.real)))
# plt.subplot(3, 1, 3)
# plt.plot(np.log(abs(f.imag)))
# plt.show()