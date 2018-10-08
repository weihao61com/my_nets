import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

DIMENSION = 2
NOISE = 0.2

def int_min_max(ymin, ymax):
    iymin = int(ymin)
    iymax = int(ymax)
    if iymin > ymin:
        iymin -= 1
    if iymax<ymax:
        iymax += 1
    #return iymin, iymax
    return -4, 4


def set_min_max():
    ymin, ymax = plt.ylim()
    xmin, xmax = plt.xlim()
    x = xmax - xmin
    y = ymax - ymin

    if x>y:
        x -= y
        ymin, ymax = int_min_max(ymin-x/2, ymax+x/2)
        xmin, xmax = int_min_max(xmin, xmax)
    else:
        y -= x
        xmin, xmax = int_min_max(xmin-y/2, xmax+y/2)
        ymin, ymax = int_min_max(ymin, ymax)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)


def create_data(view, outlier, num_points):
    #print num_points
    data = []
    #para_coff = np.random.rand(DIMENSION)*2 + 0.5
    para_offset = np.random.rand(DIMENSION)*2 - 1
    #print para_coff
    #print para_offset
    coff = np.random.rand(DIMENSION)*2 - 1
    #print coff

    for a in range(num_points):
        values= np.random.rand(DIMENSION)*2 - 1
        #while sum(values*coff)<0:
        #    values = np.random.rand(DIMENSION) * 2 - 1
        d2 = values * values
        distance = np.sqrt(np.sum(d2))
        values = values/distance + para_offset
        values = values + np.random.normal(0, NOISE, DIMENSION)
        data.append(values)

    if outlier>0:
        for a in range(int(num_points*outlier)):
            values = np.random.rand(DIMENSION) * 8 - 4
            data.append(values)

    data = np.array(data)
    if view:
        plt.scatter(data[:, 0], data[:,1], s=3)
        set_min_max()
        plt.show()

    nd = data.shape[1]
    for a in range(nd):
        mm = np.mean(data[:,a])
        data[:, a] -= mm
        para_offset[a] -= mm

    return data, np.concatenate((para_offset, coff))


if __name__ == '__main__':
    number_data = 5000
    MAX_num = 50
    filename = 'tr_c50.p'

    MIN_num = MAX_num
    num_points = random.randint(MIN_num, MAX_num)

    data = []
    for a in range(number_data):
        data.append(create_data(False, 0.0, num_points))

    with open(filename, 'w') as fp:
        pickle.dump(data, fp)