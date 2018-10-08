import datetime as dt
from multiprocessing import Process, current_process
import sys
from o2_load import *
import numpy as np


def f(name):
    print('{}: hello {} from {}'.format(
        dt.datetime.now(), name, current_process().name))
    x = 1
    for a in range(1000000):
        x = x*(a+1)
    sys.stdout.flush()


def r(tr):
    for cnt in range(len(tr)):
        np.random.shuffle(tr[cnt][0])

if __name__ == '__main__':

    tr_data = '/home/weihao/posenet/my_nets/stacknet/c2_tr.p'
    length = 2000
    trs = []
    worker_count = 2
    for _ in range(worker_count):
        trs.append(load_data(tr_data, length))

    loop = 1000
    p = False

    t0 = dt.datetime.now()
    for b in range(loop):

        if p:
            worker_pool = []
            for w in range(worker_count):
                p = Process(target=r, args=(trs[w],))
                p.start()
                worker_pool.append(p)

            for p in worker_pool:
                p.join()  # Wait for all of the workers to finish.
        else:
            for w in range(worker_count):
                for cnt in range(len(trs[w])):
                    np.random.shuffle(trs[w][cnt][0])
    print dt.datetime.now()-t0