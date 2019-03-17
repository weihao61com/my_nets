import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils

seg = 150000

def csv_line(dd):
    output = None
    for d in dd:
        if output is None:
            output = '{}'.format(d)
        else:
            output = '{},{}'.format(output, d)

    return output

filename = '/home/weihao/tmp/L/L_11.csv' #sys.argv[1]


with open(filename, 'r') as fp:
    lines = fp.readlines()

#
# rm = len(lines)%seg
# lines = lines[rm:]
#
# x = []
# y = []
# for line in lines:
#     v= map(float, line.split(','))
#     x.append(v[0])
#     y.append(v[1])
#
# win = len(lines)/seg
# x = np.array(x)
# x = x.reshape(4, seg/4, win)
# x = np.mean(x, 0)
# plt.plot(np.std(x, 0))
# plt.show()

id = 50
ls = lines[id*seg:(id+1)*seg]
x = []
y = []
for line in ls:
    v= map(float, line.split(','))
    x.append(v[0])
    y.append(v[1])

y = np.mean(np.array(y))
print 'total data', len(lines), y


nx = (seg-seg%4096)/4096
x = np.array(x[:nx*4096])
a = x.reshape(4096, nx)- np.mean(x)

plt.plot(np.mean(a, 0),'.-')
plt.show()