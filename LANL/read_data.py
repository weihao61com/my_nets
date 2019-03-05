import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os
from LANL_Utils import l_utils

filename = '/home/weihao/Downloads/train.csv'
p_file = 'p.p'

bucket = 15000
nb = int(sys.argv[1])
max_num = nb*bucket
nt = 0
x = []
y = []
datax = []
datay = []
header = None
t0 = datetime.datetime.now()

if os.path.exists(p_file):
    with open(p_file, 'r') as fp:
        data = pickle.load(fp)
    print data.shape

else:

    with open(filename, 'r') as fp:
        while nt<max_num:
            try:
                line = fp.next()
                if header is None:
                    header = line[:-1]
                    print header
                else:
                    v = map(float, line.split(','))
                    x.append(v[0])
                    y.append(v[1])
                    nt += 1
                    if nt%bucket == 0:
                        c1 = l_utils.get_core(x)
                        c2 = l_utils.get_core(y)
                        print c1,c2
                        x = []
                        y = []
                    npt = int(nt/bucket)
                    if nt%bucket==0 and npt%4000==0:
                        print npt, datetime.datetime.now() - t0
            except:
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
