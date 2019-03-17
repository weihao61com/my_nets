# import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils, HOME
import glob
import random
from sortedcontainers import SortedDict
from multiprocessing.dummy import Pool as ThreadPool


sys.path.append('{}/my_nets'.format(HOME))

from utils import Utils
#
# SEG = 10000
# CV = 5
# dct = False
# dim = 200
# threads = 2
# location = '/home/weihao/tmp/L'
# out_location = '/home/weihao/Projects/p_files/L/L_{}'


def process(cg):
    c= cg[0]
    files = cg[1]
    ids = cg[2]
    SEG = cg[3]
    dct = cg[4]
    dim = cg[5]
    out_loc = cg[6]

    data = []
    for filename in sorted(files):
        if ids[filename] == c:
            with open(filename, 'r') as fp:
                lines = fp.readlines()
            NF = int (len(lines) / 10000)
            t_scale = float(lines[0].split(',')[1])
            print 'records', c, filename, len(lines), len(lines) / SEG, NF, t_scale
            rps = np.random.randint(0, len(lines) - SEG - 1, NF)
            rps.sort()
            pr = 0
            for a in range(NF):
                #rm = rps[a] - pr
                #lines = lines[rm:]
                A = l_utils.get_features(lines[rps[a]:rps[a]+SEG], dct, dim)
                A[0] = A[0]/t_scale
                data.append(A)
                #pr = rps[a]
    print "Total data", c, len(data)
    filename = os.path.join(out_loc, 'L_{}.p'.format(c))
    with open(filename, 'w') as fp:
        pickle.dump(data, fp)


def main1(config, cnt):
    cfg = Utils.load_json_file(config)
    SEG = cfg['SEG']
    dct = cfg['dct'] > 0
    dim = cfg['dim']

    for id in range(cnt):
        out_loc = cfg['out_location'].format(HOME, id)
        if not os.path.exists(out_loc):
            os.mkdir(out_loc)

            files = glob.glob(os.path.join(cfg['location'].format(HOME), 'L_*.csv'))
            ids = l_utils.rdm_ids(files)

            cg = []
            for a in range(cfg['CV']):
                cg.append([a, files, ids, SEG, dct, dim, out_loc])
            #numbers = range(cfg['CV'])

            if cfg['threads']>1:
                pool = ThreadPool(cfg['threads'])
                results = pool.map(process, cg)
                pool.close()
                pool.join()
            else:
                for c in cg:
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

if __name__ == '__main__':
    config = 'config.json'
    main1(config, 3)