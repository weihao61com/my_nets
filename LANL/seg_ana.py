import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils

seg = 150000

filename = '/home/weihao/tmp/L/L_11.csv' #sys.argv[1]
#id = 10000000 # int(sys.argv[2])

with open(filename, 'r') as fp:
    line0 = fp.readlines()

print 'total data', len(line0)
#id = 40735392
#line0 = line0[id:id+150000]

step = int(len(line0)-seg)/100

for id in range(0, len(line0)-seg, step):
    lines = line0[id:id+seg]

    x = []
    y = []
    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])

    v = l_utils.feature_final(x, False, 50)
    y = np.mean(np.array(y))
    print id, y, np.mean(v[1:20]), np.std(v[1:20]),np.mean(v[20:]), np.std(v[20:])

plt.plot(v[1:])
plt.show()