import csv
import numpy as np
from sortedcontainers import SortedDict
from sklearn import linear_model
import os
# import utm
import pyproj
import math
from L_utils import LUtils, Sensors, Measurements, HOME
import pickle

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_location = '{}/../../datasets/al'.format(this_file_path)
wgs84=pyproj.Proj("+init=EPSG:4326")

a = 6378137
b = 6356752.3142
f = (a - b) / a
e_sq = f * (2-f)


class Truth:
    def __init__(self, row):
        illa = map(float, row)
        self.id = int(illa[0])
        self.lla = illa[1:]


class TruthData:
    def __init__(self,filename):
        d = LUtils.read_csv(filename)
        self.data = {}
        for a in d:
            self.data[int(a[0])] = a[1:]
        print 'Total truth data', len(self.data)


class TrainData:
    def __init__(self, id, K):
        self.K = K
        self.id = id
        self.collection = []

    def add_data(self, measurements):
        for m in measurements:
            mm = measurements[m]
            if len(self.collection)==self.K:
                break
            mm.append(m)
            self.collection.append(mm)

    def enough(self):
        return len(self.collection)==self.K


def gps_to_ecef_pyproj(l):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, l[0], l[1], l[2], radians=False)

    return x, y, z

def ecef_to_enu(x, y, z, lat0, lon0, h0):
    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    return xEast, yNorth, zUp


if __name__ == '__main__':
    import os
    import sys

    #data_location = "/Users/weihao/Downloads" #'/home/weihao/PY/al'
    data_set = 'training_1_category_4'
    filename = '{0}/{1}/{1}.csv'.format(data_location, data_set)
    truthfile = '{0}/{1}_result/{1}_result.csv'.format(data_location, data_set)
    sensorfile = '{0}/{1}/sensors.csv'.format(data_location, data_set)
    #  grep ",103," ../../datasets/al/training_1_category_4/training_1_category_4.csv > A103.csv
    #filename = 'A103.csv'.format(data_location)


    sensors = Sensors(sensorfile)
    truth = TruthData(truthfile)
    aircrafts, missing = LUtils.read_aircraft(filename)

    filename = '{}/../tmp/m.csv'.format(HOME)
    fp = open(filename, 'w')
    for id in aircrafts:
        if len(aircrafts[id])>5000:
            print '\t\taircraft', id, len(aircrafts[id])
            measure = Measurements(aircrafts[id], id, sensors, fp=fp)
            # measure.sync()
    fp.close()

    #
    # nt = 0
    # for id in missing:
    #     if len(missing[id])>500:
    #         print '\t\taircraft', id, len(missing[id])
    #         nt += 1
    #         measure = Measurements(missing[id], id, sensors, verbose=False)
    #         # measure.sync()
    #         measure.set_GT(truth)
    #         data = measure.greate_GT()
    #
    #         filename = 'T_{}.p'.format(id)
    #         with open(filename, 'w') as fp:
    #             pickle.dump(data, fp)
    #     #if nt>3:
    #     #    break
    #
    # nt = 0
    # for id in aircrafts:
    #     if len(aircrafts[id])>5000:
    #         print '\t\taircraft', id, len(aircrafts[id])
    #         nt += 1
    #         measure = Measurements(aircrafts[id], id, sensors, verbose=False)
    #         # measure.sync()
    #         data = measure.greate_GT()
    #
    #         filename = 'A_{}.p'.format(id)
    #         with open(filename, 'w') as fp:
    #             pickle.dump(data, fp)
    #     #if nt>3:
    #     #    break