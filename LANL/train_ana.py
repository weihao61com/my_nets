import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils

seg = 150000

def ana(lines):
    if len(lines) != seg:
        raise Exception('Wrong size {}'.format(len(lines)))

    x = []
    y = []
    for line in lines:
        v= map(float, line.split(','))
        x.append(v[0])
        y.append(v[1])
    v = l_utils.get_core(x)
    print int(np.mean(np.array(y))*1000), v

filename = sys.argv[1]
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

a = 0
while a<len(lines):
    ana(lines[a:a+seg])
    a += seg
