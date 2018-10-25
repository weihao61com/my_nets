# from o2_load import load_data
from scipy.optimize import minimize
import numpy as np
from math import sin, cos


def eular2m(c):
    c0 = cos(c[0])
    c1 = cos(c[1])
    c2 = cos(c[2])
    s0 = sin(c[0])
    s1 = sin(c[1])
    s2 = sin(c[2])
    m1 = np.array([[1, 0, 0], [0, c0, -s0], [0, s0, c0]])
    m2 = np.array([[c1, 0, s1], [0, 1, 0], [-s1, 0, c1]])
    m3 = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]])

    m = m1.dot(m2.dot(m3))

    return m

class R3_Opt:

    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
        self.length = len(d1)

    def f0(self, c):
        loss = []
        m = eular2m(c)

        for a in range(self.length):
            r1 = self.d1[a]
            r2 = m.dot(self.d2[a])
            loss.append((r1-r2).dot(r1-r2))

        return loss

    def f(self, c):
        loss = self.f0(c)

        return np.sum(loss)

    def opt(self):
        initial_guess = [0.0, 0.0, 0., 1.0]
        res = minimize(self.f, initial_guess)
        return res.x, res.fun


def load_txt(filename):
    r1 = []
    r2 = []
    nt = 0

    with open(filename, 'r') as fp:
        for line in fp.readlines():
            strs = line.split(' ')
            if len(strs)<5:
                continue
            vals = list(map(float, strs[2:]))

            r1.append(np.array(vals[0:3]))
            r2.append(np.array(vals[3:6]))
            nt += 1
        print "total data {} out of {}. ".format(len(r1), nt)
    return r1, r2



def load_data_new(filename):
    r1 = []
    r2 = []
    nt = 0

    with open(filename, 'r') as fp:
        for line in fp.readlines():
            vals = list(map(float, line.split(',')[2:]))

            r1.append(np.array(vals[0:3]))
            r2.append(np.array(vals[3:6]))
            nt += 1
        print "total data {} out of {}. ".format(len(r1), nt)
    return r1, r2


def load_data(filename, th=10):
    r1 = []
    r2 = []
    nt = 0
    cnt = 0
    inline = 0
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            vals = list(map(float, line.split(',')))
            st = 3
            inline += vals[st+1]
            cnt += vals[st]
            if vals[3]>th:
                r1.append(np.array(vals[st+2:st+5]))
                r2.append(np.array(vals[st+5:st+8]))
            nt += 1
        print "total data {} out of {}. average point {}, inline {}".format(len(r1), nt, cnt/nt, inline/nt)
    return r1, r2

def cal_medians(r1, r2):
    s0 = []
    s1 = []
    s2 = []
    sa = []
    length = len(r1)

    for a in range(length):
        s0.append(abs(r1[a][0]-r2[a][0]))
        s1.append(abs(r1[a][1]-r2[a][1]))
        s2.append(abs(r1[a][2]-r2[a][2]))
        r = np.linalg.norm(r1[a]-r2[a])
        sa.append(r)

    s = 180/np.pi

    print "medians(degree):  ", np.median(s0)*s, np.median(s1)*s, np.median(s2)*s, np.median(sa)*s


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    filename = '/home/weihao/Projects/tmp/heads_Test.csv'
    if len(sys.argv)>1:
        filename = sys.argv[1]

    r1, r2 = load_txt(filename)
    print "Total data", len(r1), len(r2)
    print "Truth", r1[0].shape

    cal_medians(r1,r2)

    loss_th = 1
    nt = 0

    while True:
        r3 = R3_Opt(r1, r2)

        vals, fun = r3.opt()


        #print vals, fun
        #print eular2m(vals), '\n'
        loss0 = r3.f([0,0,0,1])
        loss = r3.f0(vals)
        #print np.sum(loss), loss0
        mn = np.mean(loss)
        max_loss = loss[0]
        idx = 0
        for a in range(len(loss)):
            if loss[a]>max_loss:
                max_loss = loss[a]
                idx = a
        nt += 1
        if max_loss/np.mean(loss)<loss_th or nt>100:
            print 'values, fun', vals, fun
            mx = eular2m(vals)
            print mx
            print len(r1), max_loss, np.mean(loss)
            break
        del r1[idx]
        del r2[idx]

    r_new = []
    for r in r2:
        r_new.append(mx.dot(r))

    cal_medians(r1, r_new)

        # print len(r1), max_loss, np.mean(loss)

