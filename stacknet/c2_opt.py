from o2_load import load_data
from scipy.optimize import minimize
import numpy as np


class C2Opt:

    def __init__(self, data):
        self.data = data

    def f(self, c):
        loss = 0.0
        for a in range(self.data.shape[0]):
            d = self.data[a, :]
            d0 = d[0] - c[0]
            d1 = d[1] - c[1]
            dd = np.sqrt(d0 * d0 + d1 * d1) - 1
            loss += dd * dd
        return loss

    def opt(self):
        initial_guess = [0, 0]
        res = minimize(self.f, initial_guess)
        return res.x, res.fun


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = load_data('tr_25.p')
    print "Total data", len(data)
    print "Data", data[0][0].shape
    print "Truth", data[0][1].shape

    dist = []
    for a in range(len(data)):
        c2opt = C2Opt(data[a][0])
        vals, fun = c2opt.opt()
        # if np.abs(vals[0]-data[a][1][0])>0.5:

        # print a, vals, data[a][1]
        distance = (vals[0] - data[a][1][0]) * (vals[0] - data[a][1][0]) + \
                   (vals[1] - data[a][1][1]) * (vals[1] - data[a][1][1])
        dist.append(distance)
    print np.mean(dist), np.std(dist)
    #    plt.plot(data[a][0][:, 0], data[a][0][:, 1], '.')
    #    plt.show()
