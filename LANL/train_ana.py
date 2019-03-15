import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils

SEG = 150000

def ana(lines, a):
    if len(lines) != SEG:
        raise Exception('Wrong size {}'.format(len(lines)))

    x = []
    y = []
    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])
    # v = l_utils.get_core(x)
    v = l_utils.feature_final(x, False, 5000)
    x = np.array(range(5001))-3000
    z = np.polyfit(x[1000:], v[1000:], 2)
    #plt.plot(v[1000:])
    #plt.show()
    z1 = np.mean(v[1:1000])
    z2 = np.std(v[1:1000])
    print np.mean(np.array(y)), [z1,z2]+list(z), a

filename = '/home/weihao/tmp/L/L_12.csv' #sys.argv[1]
filename = sys.argv[1]
max_seg = 200

with open(filename, 'r') as fp:
    lines = fp.readlines()

step = int(len(lines)/max_seg)
#print 'total data', len(lines), step

a = 0
while a+SEG<len(lines):
    ana(lines[a:a+SEG], a)
    a += step
