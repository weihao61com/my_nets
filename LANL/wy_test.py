import os
import glob
from scipy import fftpack, fft
import numpy as np
import matplotlib.pyplot as plt
import sys
import pywt

NF = 200
level = 8
SEG = 4096
#STEP = 5000
cl = ['r','g','b']

def do_fft(x):
    length = len(x)
    win = length/2
    while win>32:
        win /= 2
    d = abs(fft(x))
    dv = np.array(d[1:length/2+1]).reshape(win, length/win/2)
    dv = np.mean(dv, 1)
    return dv

def run_file(file):
    with open(file, 'r') as fp:
        lines = fp.readlines()
    #fig, ax = plt.subplots()

    # print 'total', len(lines) / SEG
    step = (len(lines) - SEG - 1) / NF

    v = map(float, lines[0].split(','))
    my = v[1]
    nt = 0
    a1 = 6
    a2 = 4
    for a in range(NF): #NF-3):
        if a>2 and a<NF-3:
            continue
        s = 0
        if a>5:
            s=1

        p = a * step
        x = []
        y = []
        for line in lines[-(p + SEG+1):-p-1]:
            v = map(float, line.split(','))
            x.append(v[0])
            y.append(v[1])

        ny = np.mean(y)



        c = pywt.wavedec(x, 'db1', level=level)
        #if np.std(c[level])>0:

        print ny/my, ny, np.mean(abs(c[0])),
        for l in range(level):
            print '{0:6.3f}'.format(np.std(c[l + 1])),

        # cv = np.std(c[level])
        for a in range(len(c)):
            c[a] = do_fft(c[a])

        plt.subplot(a1, a2, 1+s)
        plt.plot(x)
        plt.subplot(a1, a2, 3+s)
        plt.plot(c[level])
        nt+=1
        plt.subplot(a1, a2, 5+s)
        plt.plot(c[level-1])
        plt.subplot(a1, a2, 7+s)
        plt.plot(c[level-2])
        plt.subplot(a1, a2, 9+s)
        plt.plot(c[level-3])
        plt.subplot(a1, a2, 11+s)
        plt.plot(c[level - 4])
        plt.subplot(a1, a2, 13+s)
        plt.plot(c[level - 5])
        plt.subplot(a1, a2, 15+s)
        plt.plot(c[level - 6])
        plt.subplot(a1, a2, 17+s)
        plt.plot(c[level - 7])
        plt.subplot(a1, a2, 19+s)
        plt.plot(c[level - 8])
        #plt.show()

        print

    #leg = plt.legend()
    #plt.legend(loc='upper left', frameon=False)
    #da = np.array(da)
    plt.show()
    print


id = -1
if len(sys.argv)>1:
    id = int(sys.argv[1])

location = '/home/weihao/tmp/L' #sys.argv[1]
files = glob.glob(os.path.join(location, 'L_*.csv'))

if id>-1:
    run_file(files[id])
else:
    for f in files:
        run_file(f)

