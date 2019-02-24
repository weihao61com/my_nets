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
from bluenotelib.common.coordinate_transforms import CoordinateTransforms as ct


this_file_path = os.path.dirname(os.path.realpath(__file__))
data_location = '{}/../../datasets/al'.format(this_file_path)
output_location = '{}/../../p_files'.format(this_file_path)
#wgs84=pyproj.Proj("+init=EPSG:4326")

#a = 6378137
#b = 6356752.3142
#f = (a - b) / a
#e_sq = f * (2-f)


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


def gps_to_ecef_pyproj_deprecated(l):
    xyz = np.array(ct.geodetic_to_ecef(l[1], l[0], l[2]))
    return xyz[0], xyz[1], xyz[2]


# def ecef_to_enu(x, y, z, lat0, lon0, h0):
#     lamb = math.radians(lat0)
#     phi = math.radians(lon0)
#     s = math.sin(lamb)
#     N = a / math.sqrt(1 - e_sq * s * s)
#
#     sin_lambda = math.sin(lamb)
#     cos_lambda = math.cos(lamb)
#     sin_phi = math.sin(phi)
#     cos_phi = math.cos(phi)
#
#     x0 = (h0 + N) * cos_lambda * cos_phi
#     y0 = (h0 + N) * cos_lambda * sin_phi
#     z0 = (h0 + (1 - e_sq) * N) * sin_lambda
#
#     xd = x - x0
#     yd = y - y0
#     zd = z - z0
#
#     xEast = -sin_phi * xd + cos_phi * yd
#     yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
#     zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd
#    return xEast, yNorth, zUp

def save_data(data, filenm):
    print 'save data', filenm
    with open(filenm, 'w') as fp:
        pickle.dump(data, fp)


def sync(aircrafts, sensors):
    sensor_list = SortedDict()
    for id in aircrafts:
        self = Measurements(aircrafts[id], id, sensors, fp=None)
        for t in self.data:
            dd = self.data[t]
            for m in dd.measurements:
                if not m in sensor_list:
                    sensor_list[m] = []
                mm = dd.measurements[m]
                sensor_list[m].append((dd.timeAtServer, mm[0] * 1e-9))

    for m in sensor_list:
        if len(sensor_list[m]) > 10:
            z = LUtils.l_fit(sensor_list[m])
            print 'sync', m, len(sensor_list[m]), z[0], z[1]

def read_sensor_sync(filename):
    ss = {}
    for v in LUtils.read_csv(filename):
        ss[v[0]] = float(v[1])
    return ss


if __name__ == '__main__':
    import os
    import sys

    data_location = "/Volumes/WEI/AC_Localization" #'/home/weihao/PY/al'
    data_set = 'training_2_category_4'
    filename = '{0}/{1}/{1}.csv'.format(data_location, data_set)
    truthfile = '{0}/{1}_result/{1}_result.csv'.format(data_location, data_set)
    sensorfile = '{0}/{1}/sensors.csv'.format(data_location, data_set)
    sensor_sync = 'sensor_sync.csv'
    #  grep ",103," ../../datasets/al/training_1_category_4/training_1_category_4.csv > A103.csv
    #filename = 'A512.csv'.format(data_location)
    max_rec = 1e6

    sync = read_sensor_sync(sensor_sync)
    sensors = Sensors(sensorfile)
    truth = TruthData(truthfile)
    aircrafts, missing = LUtils.read_aircraft(filename)
    print len(aircrafts), len(missing)


    filename = '{}/../tmp/m.csv'.format(HOME)
    data = {}
    nf = 0
    r1 = 0
    r2 = 0
    fp = None #open(filename, 'w')
    for id in aircrafts:
        r1 += len(aircrafts[id])

        if len(aircrafts[id])>1000:
            r2 += len(aircrafts[id])
            print '\t\taircraft', id, len(aircrafts[id]), r2, len(data)
            measure = Measurements(aircrafts[id], id, sensors, fp=fp)
            data[id] = measure.outputs
            if r2 > max_rec:
                output_file = '{0}/{1}_{2}.p'.format(output_location, data_set, nf)
                save_data(data, output_file)
                nf += 1
                data = {}
                r2 = 0

            #measure.sync()
    if fp:
        fp.close()
    print 'recoeds', r1,r2
    if len(data)>0:
        output_file = '{0}/{1}_{2}.p'.format(output_location, data_set, nf)
        save_data(data, output_file)
