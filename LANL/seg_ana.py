import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils

seg = 150000

filename = sys.argv[1]
id = int(sys.argv[2])
max_seg = 130

with open(filename, 'r') as fp:
    lines = fp.readlines()

nseg = len(lines)/seg
print 'total data', len(lines)

if len(lines)>max_seg*seg:
    rm = -seg*max_seg
else:
    rm = len(lines)%seg
lines = lines[rm:]
nseg = len(lines)/seg
print 'total data', len(lines), nseg

lines = lines[id*seg:(id+1)*seg]

x = []
y = []
for line in lines:
    v= map(float, line.split(','))
    x.append(v[0])
    y.append(v[1])

sub = 5000
fm = []
fs = []
a = 0
while a < len(x):
    v = l_utils.get_core(x[a:a+sub])
    fs.append(v[1])
    fm.append(v[0])
    a += sub

y = np.mean(np.array(y))
print y
for a in range(len(fs)):
    print fs[a], ',', fm[a]
plt.plot(x)
plt.show()