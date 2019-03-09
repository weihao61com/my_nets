import numpy as np
import matplotlib.pyplot as plt


class BinData():
    def __init__(self):
        self.data = []

    def add(self, point):
        self.data.append(point)

    def get_V(self):
        return sum(self.data)

    def get_C(self):
        return len(self.data)


class RN:
    def __init__(self, limit=100, level=0):
        self.th = None
        self.l = None
        self.r = None
        self.limit = limit
        self.level = level

        self.value = None
        self.std = None
        self.cnt = None

    def get_v(self, d):
        if self.th is None:
            v = self.value
        elif self.left(d[1]):
            v = self.l.get_v(d)
        else:
            v = self.r.get_v(d)
        return v

    def predict(self, dd):
        output = []
        for d in dd:
            output.append(self.get_v(d))
        return output

    def fit(self, X, y):
        if len(y)<self.limit or self.level>20:
            self.value = np.mean(y)
            self.std = np.std(y)
            self.cnt = len(y)
        else:
            self.id, self.th = self.split(X, y)
            self.l = RN(self.limit, self.level+1)
            self.r = RN(self.limit, self.level+1)
            X1 = []
            X2 = []
            Y1 = []
            Y2 = []
            for a in range(len(y)):
                if self.left(X[a][1]):
                    X1.append(X[a])
                    Y1.append(y[a])
                else:
                    X2.append(X[a])
                    Y2.append(y[a])
            # print 'fit',self.id, self.level, len(Y1), len(Y2)

            self.l.fit(X1, Y1)
            self.r.fit(X2, Y2)

    def left(self, x):
        return x[self.id]<self.th

    def get_random_isd(self, dim, prob=0.03):
        ids = range(dim)
        np.random.shuffle(ids)
        return ids[:int(dim*prob)]

    def get_th(self, dd, id0, ch=100):

        # plt.subplot(3, 1, 3)
        # plt.plot(x, y, '.')
        # plt.show()
        x = []
        y = []
        for d in dd:
            y.append(d[0])
            x.append(d[1][id0])

        v0 = np.min(x)
        v1 = np.max(x)
        # print id0, v0, v1
        if v1==v0:
            print 'No split'
            return None

        dv = (v1-v0)/ch
        points = {}
        for a in range(ch):
            points[a] = BinData()

        chs = np.floor((x-v0)/dv).astype(int)

        for a in range(len(chs)):
            id = chs[a]
            if id==ch:
                id = ch-1
            points[id].add(y[a])

        # print len(points)
        s1 = 0.0
        n1 = 0.0
        for id in points:
            if points[id].get_C()>0:
                # print id, points[id].get_V(),points[id].get_C()
                s1 += points[id].get_V()*points[id].get_C()
                n1 += points[id].get_C()

        # print s1, n1
        total = n1
        s2 = 0.0
        n2 = 0.0
        MS = -1
        M_id = 0
        for id in points:
            if points[id].get_C()>0:
                s1 -= points[id].get_V()*points[id].get_C()
                n1 -= points[id].get_C()
                s2 += points[id].get_V()*points[id].get_C()
                n2 += points[id].get_C()
                if n1>0:
                    x1 = s1/n1
                    x2 = s2/n2
                    S = ((total-n1)*n2)/total/total*(x1-x2)
                    T = np.sqrt((total-n1)*n2/total/total)*(x1-x2)
                    if MS < abs(T):
                        MS = abs(T)
                        M_id = id
                        #print id0, M_id, S, T, x1, x2, n1, n2
        th = (M_id + 1)*dv + v0

        return MS, th

    def split(self, X, y):
        ids = self.get_random_isd(len(X[0][1]))
        MS = -1
        M_id = None
        for id in ids:
            ms, th = self.get_th(X, id)
            if ms>MS:
                MS = ms
                M_id = (id, th)

        return M_id

    def pridict(self, y):
        return 0