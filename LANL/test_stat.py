import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
import glob
from LANL_Utils import l_utils


def get_stat(filename):
    data = []
    header = None
    with open(filename, 'r') as fp:
        while True:
            try:
                line = fp.next()
                if header is None:
                    header = line[:-1]
                    # print header
                else:
                    data.append(float(line[:-1]))
            except:
                break
    v = l_utils.get_core(data)

    return v


location = '/home/weihao/Downloads/test'

files = glob.glob(os.path.join(location, '*.csv'))
print 'Total file', len(files)

for f in files:
    stat = get_stat(f)
    print f, stat
