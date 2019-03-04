import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pickle
import os

filename = '/home/weihao/Downloads/train.csv'
p_file = 'p.p'

bucket = 15000
nb = int(sys.argv[1])
max_num = nb*bucket
nt = 0
x = []
y = []
data = []
header = None
t0 = datetime.datetime.now()

if os.path.exists(p_file):
    with open(p_file, 'r') as fp:
        data = pickle.load(fp)

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
                        p = []
                        p.append(np.mean(x))
                        p.append(np.mean(y))
                        p.append(np.std(x))
                        p.append(np.std(y))
                        data.append(p)
                        x = []
                        y = []
                    npt = int(nt/bucket)
                    if nt%bucket==0 and npt%1000==0:
                        print npt, datetime.datetime.now() - t0
            except:
                break
    data = np.array(data)

    with open('p.p', 'w') as fp:
        pickle.dump(data, fp)

nd = []
for d in data:
    if d[2]<40:
        nd.append(d)

data = np.array(nd)


for a in range(4):
    plt.subplot(2, 2, a+1)
    plt.plot(data[:nb, a])

plt.show()
