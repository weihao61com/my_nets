import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils

filename = '/home/weihao/Downloads/train.csv'
output_location = '/home/weihao/tmp/L_{}.csv'
header = None
pre_t = None
max_num = 1e9
nt = 0
seg = 0
t0 = datetime.datetime.now()
data = []
fx = open(output_location.format(seg), 'w')

with open(filename, 'r') as fp:
    while True:
        try:
            line = fp.next()
            if header is None:
                header = line[:-1]
                print header
            else:
                v = map(float, line.split(','))
                if pre_t is None:
                    pre_t = v[1]
                dt = v[1] - pre_t
                if dt>0:
                    print seg, nt, pre_t, v, datetime.datetime.now()-t0
                    fx.close()
                    seg += 1
                    fx = open(output_location.format(seg), 'w')
                fx.write(line)

                pre_t = v[1]
                nt += 1

        except Exception as e:
            print e.message
            break

# with open('p.p', 'w') as fp:
#    pickle.dump(data, fp)



# st=10000
#
# for a in range(3):
#     plt.subplot(2, 2, a+1)
#     plt.plot(data[st:nb, a])
#
# plt.subplot(2, 2, 4)
# plt.plot(data[st:nb, 1], data[st:nb, 2], '.')
#
# plt.show()
