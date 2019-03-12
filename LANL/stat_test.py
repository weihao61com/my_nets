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

filename = sys.argv[1]


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

nseg = len(lines)/seg

id = 250
ls = lines[id*seg:(id+1)*seg]
x = []
y = []
for line in ls:
    v= map(float, line.split(','))
    x.append(v[0])
    y.append(v[1])

y = np.mean(np.array(y))
print 'total data', len(lines), y, nseg

ch = 10
x = np.array(x)
xv = x.reshape(ch, seg/ch)
plt.plot(x)
fig, ax = plt.subplots()

for a in range(ch):
    f = abs(np.fft.fft(xv[a, :]))
    f0 = f[0]
    f = f[1:seg/ch/2+1]
    f = f.reshape(15,500)
    f = np.mean(f, 1)
    ax.plot(f[4:], c='{}'.format(float(a)/ch), label='{}'.format(a))
    print a, f0, np.mean(f), np.mean(f[4:])

leg = ax.legend();
ax.legend(loc='upper right', frameon=False)

plt.show()