import cv2
import numpy as np
import os
import random
import json
from sortedcontainers import SortedDict
import sys
import pickle
import shutil
import csv

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'


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
            self.gAlt = float(row[6])
        self.numMeasurements = int(row[7])
        newstr = row[8].replace("]", "").replace("[", "").split(',')
        mtx = np.array(map(float, newstr)).reshape((self.numMeasurements, 3))
        self.measurements = {}
        for a in range(self.numMeasurements):
            self.measurements[int(mtx[a, 0])] = list(mtx[a, 1:])

    def add_sensor(self, s):
        if s.id in self.measurements:
            self.measurements[s.id] = self.measurements[s.id] + s.lla


class Sensor:
    def __init__(self, rows):
        self.id = int(rows[0])
        self.lla = map(float, rows[1:-1])
        self.name = rows[-1]
        self.type_id = None
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


class Measurements:

    def __init__(self, rows, a, sensors, verbose=True):
        self.data = SortedDict()
        self.sensors = sensors
        self.a = a

        for row in rows:
            d = Data(row)
            self.data[d.id] = d

        print 'total aircraft data', a, len(self.data)

        # if verbose:
        #     filename = '{}/m_{}.csv'.format(this_file_path, a)
        #     with open(filename, 'w') as fp:
        #         for t in self.data:
        #             dd = self.data[t]
        #             for m in dd.measurements:
        #                 mm = dd.measurements[m]
        #                 ss = sensors.data[m]
        #                 lla = ss.lla
        #                 xy = ss.xyz
        #                 xyz = ecef_to_enu(dd.lat, dd.lon, dd.bAlt, lla[0],lla[1],lla[2])
        #                 fp.write('{},{}'.format(dd.timeAtServer, dd.id))
        #                 fp.write(',{},{},{},{},{},{},{}'.
        #                          format(xyz[0], xyz[1], xyz[2], dd.lat, dd.lon, dd.bAlt, dd.gAlt))
        #                 fp.write(',{},{},{},{}'.format(m, ss.type_id, mm[0], mm[1]))
        #                 fp.write(',{},{},{},{},{}'.
        #                          format(xy[0], xy[1], lla[0], lla[1], lla[2]))
        #                 r1 = np.sqrt((xyz[0]-xy[0])*(xyz[0]-xy[0])+(xyz[1]-xy[1])*(xyz[1]-xy[1]))
        #                 r2 = np.sqrt((dd.lat-lla[0])*(dd.lat-lla[0])+(dd.lon-lla[1])*(dd.lon-lla[1]))
        #                 fp.write(',{},{}'.format(r1,r2))
        #                 fp.write('\n')
        #                 #if m==514:
        #                 #    print xyz
        #     exit()

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
                truth.append([dd.lat, dd.lon, dd.bAlt])
        print 'total memsurement', len(seq)
        return seq, truth


class LUtils:

    @staticmethod
    def read_csv(filename):
        data = []
        with open(filename, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                if 'latitude' in row:
                    print '\t\t\t', row
                else:
                    data.append(row)
        # print 'Total csv data', len(data)
        return data

    @staticmethod
    def read_aircraft(filename):
        aircrafts = SortedDict()
        missing = 0

        for row in LUtils.read_csv(filename):

            d = Data(row)
            if d.missing:
                missing += 1
            else:
                if not d.aircraft in aircrafts:
                    aircrafts[d.aircraft] = []
                aircrafts[d.aircraft].append(row)

        print 'total aircraft', len(aircrafts), missing

        return aircrafts

    @staticmethod
    def l_fit(data):

        d = np.array(data)
        X = d[:, 0]
        Y = d[:, 1]
        z = np.polyfit(X, Y, 1)
        return z
