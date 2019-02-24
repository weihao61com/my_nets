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

    rst = {}
    for m in sensor_list:
        if len(sensor_list[m]) > 10:
            z = LUtils.l_fit(sensor_list[m])
            print 'sync', m, len(sensor_list[m]), z[0], z[1]
            rst[m] = [len(sensor_list[m]), z[0], z[1]]
    return rst

def w_average(data):
    x = []
    w = []
    for d in data:
        x.append(d[1])
        w.append(d[0]*d[0])

    s = np.average(x, 0, w)
    return s

if __name__ == '__main__':
    import os
    import sys

    data_location = "/Volumes/WEI/AC_Localization" #'/home/weihao/PY/al'
    data_sets = ['training_1_category_4', 'training_2_category_4']
    rst = {}

    for data_set in data_sets:
        filename = '{0}/{1}/{1}.csv'.format(data_location, data_set)
        truthfile = '{0}/{1}_result/{1}_result.csv'.format(data_location, data_set)
        sensorfile = '{0}/{1}/sensors.csv'.format(data_location, data_set)

        sensors = Sensors(sensorfile)
        aircrafts, missing = LUtils.read_aircraft(filename)
        print data_set, len(aircrafts), len(missing)
        sen = sync(aircrafts, sensors)
        for m in sen:
            if m not in rst:
                rst[m] = []
            rst[m].append(sen[m])

    with open('sensor_sync.csv', 'w') as fp:
        for m in rst:
            s = w_average(rst[m])
            print m, s, rst[m]
            fp.write('{},{}\n'.format(m,s))
