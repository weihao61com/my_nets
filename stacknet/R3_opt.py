from o2_load import load_data
from scipy.optimize import minimize
import numpy as np
from math import sin, cos


class R3_Opt:

    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
        self.length = len(d1)

    def f(self, c):
        loss = 0.0

        c0 = cos(c[0])
        c1 = cos(c[1])
        c2 = cos(c[2])
        s0 = sin(c[0])
        s1 = sin(c[1])
        s2 = sin(c[2])

        m1 = np.array([[1, 0 ,0], [0, c0, -s0], [0, s0, c0]])
        m2 = np.array([[c1, 0 ,s1], [0, 1, 0], [-s1, 0, c1]])
        m3 = np.array([[c2, -s2 ,0], [s2, c2, 0], [0, 0, 1]])
        m = m1*m2*m3

        for a in range(self.length):
            r1 = self.d1[a]
            r2 = c[3] * m.dot(self.d2[a])
            loss += (r1-r2).dot(r1-r2)

        return loss

    def opt(self):
        initial_guess = [0, 0, 0.2, 1]
        res = minimize(self.f, initial_guess)
        return res.x, res.fun

def load_data(filename):
    r1 = []
    r2 = []
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            vals = list(map(float, line.split(' ')))
            r1.append(np.array(vals[3:6]))
            r2.append(np.array(vals[6:9]))
    return r1, r2

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    filename = '/home/weihao/Downloads/cv_gc.txt'
    r1, r2 = load_data(filename)
    print "Total data", len(r1), len(r2)
    print "Truth", r1[0].shape

    r3 = R3_Opt(r1, r2)

    vals, fun = r3.opt()

    print vals
    print fun

