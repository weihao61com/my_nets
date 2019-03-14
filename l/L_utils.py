import cv2
import numpy as np
import os
import random
import json
from sortedcontainers import SortedDict
import sys
import pickle
import pyproj
import csv
from bluenotelib.common.coordinate_transforms import CoordinateTransforms as ct


HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

SPEED = 299792458.0

class Data:
    def __init__(self, row):
        self.id = int(row[0])
        self.timeAtServer = float(row[1])
        self.aircraft = int(row[2])
        self.missing = True
        if row[3] is not '':
            self.missing = False
            self.lat = float(row[3])
            self.lon = float(row[4])
            self.bAlt = float(row[5])
            self.alt = float(row[6])
            self.xyz = np.array(ct.geodetic_to_ecef(self.lon, self.lat, self.alt))

        self.numMeasurements = int(row[7])
        newstr = row[8].replace("]", "").replace("[", "").split(',')
        mtx = np.array(map(float, newstr)).reshape((self.numMeasurements, 3))
        sz = mtx.shape
        self.measurements = np.concatenate((np.array([self.timeAtServer]), mtx.reshape(sz[0]*sz[1])))
        #for a in range(self.numMeasurements):
        #    self.measurements[int(mtx[a, 0])] = list(mtx[a, 1:])

    def add_sensor(self, s):
        if s.id in self.measurements:
            self.measurements[s.id] = self.measurements[s.id] + s.lla


class Sensor:
    def __init__(self, rows):
        self.id = int(rows[0])
        l = map(float, rows[1:-1])
        self.lla = l
        self.name = rows[-1]
        self.type_id = None
        self.xyz = np.array(ct.geodetic_to_ecef(l[1], l[0], l[2]))
        # try:
        #     self.xyz = gps_to_ecef_pyproj(self.lla)
        # except Exception as e:
        #     print e.message
        #     print rows
        #self.xyz = utm.from_latlon(self.lla[0], self.lla[1])


class Sensors:
    def __init__(self, filename):
        self.data = {}
        self.ss = {}
        ns = 0
        for row in LUtils.read_csv(filename):
            s = Sensor(row)
            self.data[s.id] = s
            if s.name not in self.ss:
                self.ss[s.name] = ns
                ns += 1
        print 'Total Sensor', len(self.data), 'type', len(self.ss)

        for id in self.data:
            dd = self.data[id]
            dd.type_id = self.ss[dd.name]

def get_GT(filename):
    with open(filename, 'r') as fp:
        data = pickle.load(fp)
    return data


class Measurements:

    def __init__(self, rows, a, sensors, fp=None):
        self.data = SortedDict()
        self.sensors = sensors
        self.a = a
        self.outputs = SortedDict()

        for row in rows:
            d = Data(row)
            truth = np.array([d.lat, d.lon, d.alt])
            self.outputs[d.id] = (d.measurements, truth)
            self.data[d.id] = d

        #print 'total aircraft data', a, len(self.data)

        if fp is not None:
            counts = {}
            #with open(filename, 'w') as fp:
            D0 = None
            pt = 1000

            for t in self.data:
                dd = self.data[t]
                for m in dd.measurements:
                    if m not in counts:
                        counts[m] = 0
                    counts[m] += 1
                for id1 in dd.measurements:
                    for id2 in dd.measurements:
                        if id1>=id2:
                            continue

                        lla = [dd.lat, dd.lon, dd.alt]
                        m1 = dd.measurements[id1]
                        s1 = sensors.data[id1]
                        lla1 = s1.lla
                        m2 = dd.measurements[id2]
                        s2 = sensors.data[id2]
                        lla2 = s2.lla
                        if not pt==dd.timeAtServer:
                            dt1 = (m1[0] - m2[0]) * 1e-9
                            dif = LUtils.gps_to_ecef_pyproj(lla)
                            d1 = np.linalg.norm(dif - LUtils.gps_to_ecef_pyproj(lla1))
                            d2 = np.linalg.norm(dif - LUtils.gps_to_ecef_pyproj(lla2))
                            # d1 = LUtils.cal_dist(lla, lla1)
                            # d2 = LUtils.cal_dist(lla, lla2)
                            # dt2 = (m2[0]-m3[0])*1e-9
                            # dt3 = (m3[0]-m1[0])*1e-9
                            dif = dt1 * SPEED - (d1 - d2)
                            if D0 is None:
                                D0 = dif
                                #print 'D0', a, D0

                            #if abs(dif-36000)<1000 and abs(dd.lat-49.5)<0.5 and abs(dd.lon-8.5)<0.5:
                            fp.write('{},{},{}'.format(dd.timeAtServer, dd.id, a))
                            fp.write(',{},{},{},{}'.format( dd.lat, dd.lon, dd.bAlt, dd.alt))
                            fp.write(',{},{},{},{}'.format(id1, s1.type_id, m1[0], m1[1]))
                            # fp.write(',{},{},{}'.format(lla1[0], lla1[1], lla1[2]))
                            fp.write(',{},{},{},{}'.format(id2, s2.type_id, m2[0], m2[1]))
                            # fp.write(',{},{},{}'.format(lla2[0], lla2[1], lla2[2]))
                            #fp.write(',{},{},{}'.format(s3.type_id, m3[0], m3[1]))

                            fp.write(',{},{},{},{}'.format(dt1, d1, d2, dif))
                            fp.write('\n')
                            pt = dd.timeAtServer

            #for m in counts:
            #    if counts[m]>1000:
            #        print m, counts[m], sensors.data[m].type_id

    def sync(self):
        sensor_list = SortedDict()
        for t in self.data:
            dd = self.data[t]
            for m in dd.measurements:
                if not m in sensor_list:
                    sensor_list[m] = []
                mm = dd.measurements[m]
                sensor_list[m].append((dd.timeAtServer, mm[0]*1e-9 ))

        for m in sensor_list:
            if len(sensor_list[m])>10:
                z = LUtils.l_fit(sensor_list[m])
                print 'sync', m, self.a, len(sensor_list[m]), z[0], z[1]

    def greate_GT(self):
        seq = []
        truth = []
        for t in self.data:
            dd = self.data[t]
            for sid in dd.measurements:
                mm = dd.measurements[sid]
                s = self.sensors.data[sid]
                d = [dd.timeAtServer, mm[1], s.type_id]
                d = d + s.lla
                seq.append(d)
                truth.append([dd.lat, dd.lon, dd.gAlt])
        print 'total measurement', len(seq)
        return seq, truth

    def set_GT(self, truth):
        for id in self.data:
            tr = truth.data[id]
            self.data[id].lat = float(tr[0])
            self.data[id].lon = float(tr[1])
            self.data[id].gAlt = float(tr[2])


class LUtils:

    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    @staticmethod
    def gps_to_ecef_pyproj(l):
        # ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        # lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        # # lonlat
        x, y, z = pyproj.transform(LUtils.lla, LUtils.ecef, l[1], l[0], l[2], radians=False)
        return np.array([x, y, z])

        #xyz = np.array(ct.geodetic_to_ecef(l[1], l[0], l[2]))
        #return xyz

    @staticmethod
    def cal_dist(lla1, lla2):
        l1 = [lla1[1], lla1[0], lla1[2]]
        l2 = [lla2[1], lla2[0], lla2[2]]
        enu = ct.lla_to_enu_from_lonlatalt(l1, l2)
        return np.linalg.norm(enu)

    @staticmethod
    def read_aircraft(filename):
        aircrafts = SortedDict()
        missing_a = SortedDict()
        missing = 0

        for row in LUtils.read_csv(filename):

            d = Data(row)
            if d.missing:
                missing += 1
                if not d.aircraft in missing_a:
                    missing_a[d.aircraft] = []
                missing_a[d.aircraft].append(row)
            else:
                if not d.aircraft in aircrafts:
                    aircrafts[d.aircraft] = []
                aircrafts[d.aircraft].append(row)

        print 'total aircraft', len(aircrafts), missing,len(missing_a)

        return aircrafts, missing_a

    @staticmethod
    def l_fit(data):

        d = np.array(data)
        X = d[:, 0]
        Y = d[:, 1]
        z = np.polyfit(X, Y, 1)
        return z


if __name__ == '__main__':
    import os
    import sys

    data_location = "/Users/weihao/Projects/datasets/al/" #'/home/weihao/PY/al'
    data_set = 'training_1_category_4'
    filename = '{0}/{1}/{1}.csv.p'.format(data_location, data_set)

    data = get_GT(filename)
    print len(data)