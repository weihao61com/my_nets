import numpy as np


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
    def __init__(self, limit=1000, level=0):
        self.th = None
        self.l = None
        self.r = None
        self.limit = limit
        self.level = level

        self.value = None
        self.std = None
        self.cnt = None

    def fit(self, X, y):
        if len(y)<self.limit:
            self.value = np.mean(y)
            self.std = np.std(y)
            self.cnt = len(y)
        else:
            self.th = self.get_th(X, y)

    def get_random_isd(self, dim, prob=0.3):
        ids = range(dim)
        np.random.shuffle(ids)
        return ids[:int(dim*prob)]

    def split(self, x, y, ch=100):

        v0 = np.min(x)
        v1 = np.max(x)
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

        print len(points)
        s1 = 0.0
        n1 = 0.0
        for id in points:
            if points[id].get_C()>0:
                s1 += points[id].get_V()*points[id].get_C()
                n1 += points[id].get_C()

        print s1, n1
        total = n1
        s2 = 0.0
        n2 = 0.0
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
                    print id, S, T, x1, x2


        print ''

    def get_th(self, X, y):
        ids = self.get_random_isd(X.shape[1])
        for id in ids:
            A = self.split(X[:, id], y)

        return None

    def pridict(self, y):
        return 0