import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os

filename = '/Users/weihao/Downloads/test/seg_0cca14.csv'

step = 409
nt = 0
data = []
x = []
header = None
t0 = datetime.datetime.now()


with open(filename, 'r') as fp:
    while True:
        try:
            line = fp.next()
            if header is None:
                header = line[:-1]
                print header
            else:
                v = map(float, line.split(','))
                x.append(v[0])
                nt += 1
                if nt%step == 0:
                    x = np.array(x)
                    data.append([np.mean(x), np.std(x)])
                    x = []
        except:
            break
data = np.array(data)


for a in range(2):
    plt.subplot(2, 1, a+1)
    plt.plot(data[:, a])
plt.show()
