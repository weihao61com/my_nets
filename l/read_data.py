import csv
import numpy as np
from sortedcontainers import SortedDict
from sklearn import linear_model
import os
# import utm
import pyproj
import math

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
            if int(mtx[a, 0])==521:
                print row

    def add_sensor(self, s):
        if s.id in self.measurements:
            self.measurements[s.id] = self.measurements[s.id] + s.lla


class TruthData:
    def __init__(self,filename):
        self.header = None
        self.data = []
        with open(filename, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                if self.header is None:
                    print row
                    self.header = row
                else:
                    self.data.append(Truth(row))
        print 'Total Truth', len(self.data)

def l_fit(data):

    d = np.array(data)
    X = d[:, 0]
    Y = d[:, 1]
    z = np.polyfit(X, Y, 1)
    return z

class Measurements:

    def __init__(self, rows, a, sensors, verbose=True):
        self.data = SortedDict()
        self.sensors = sensors
        self.a = a

        for row in rows:
            d = Data(row)
            self.data[d.id] = d

        print 'total data', a, len(self.data)

        if verbose:
            filename = '{}/m_{}.csv'.format(this_file_path, a)
            with open(filename, 'w') as fp:
                for t in self.data:
                    dd = self.data[t]
                    for m in dd.measurements:
                        mm = dd.measurements[m]
                        ss = sensors.data[m]
                        lla = ss.lla
                        xy = ss.xyz
                        xyz = ecef_to_enu(dd.lat, dd.lon, dd.bAlt, lla[0],lla[1],lla[2])
                        fp.write('{},{}'.format(dd.timeAtServer, dd.id))
                        fp.write(',{},{},{},{},{},{},{}'.
                                 format(xyz[0], xyz[1], xyz[2], dd.lat, dd.lon, dd.bAlt, dd.gAlt))
                        fp.write(',{},{},{},{}'.format(m, ss.type_id, mm[0], mm[1]))
                        fp.write(',{},{},{},{},{}'.
                                 format(xy[0], xy[1], lla[0], lla[1], lla[2]))
                        r1 = np.sqrt((xyz[0]-xy[0])*(xyz[0]-xy[0])+(xyz[1]-xy[1])*(xyz[1]-xy[1]))
                        r2 = np.sqrt((dd.lat-lla[0])*(dd.lat-lla[0])+(dd.lon-lla[1])*(dd.lon-lla[1]))
                        fp.write(',{},{}'.format(r1,r2))
                        fp.write('\n')
                        #if m==514:
                        #    print xyz
            exit()

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
                z = l_fit(sensor_list[m])
                print 'sync', m, self.a, len(sensor_list[m]), z[0], z[1]


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

class Sensor:
    def __init__(self, rows):
        self.id = int(rows[0])
        self.lla = map(float, rows[1:-1])
        self.name = rows[-1]
        self.type_id = None
        try:
            self.xyz = gps_to_ecef_pyproj(self.lla)
        except Exception as e:
            print e.message
            print rows
        #self.xyz = utm.from_latlon(self.lla[0], self.lla[1])


class Sensors:
    def __init__(self, filename):
        self.header = None
        self.data = {}
        self.ss = {}
        ns = 0
        with open(filename, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                if self.header is None:
                    print row
                    self.header = row
                else:
                    s = Sensor(row)
                    self.data[s.id] = s
                    if s.name not in self.ss:
                        self.ss[s.name] = ns
                        ns += 1
        print 'Total Sensor', len(self.data), 'type', len(self.ss)

        for id in self.data:
            dd = self.data[id]
            dd.type_id = self.ss[dd.name]




def read_aircraft(filename):
    aircrafts = SortedDict()
    missing = 0
    header = None

    with open(filename, 'r') as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        for row in csv_reader:
            if header is None:
                header = row
                print row
            else:
                d = Data(row)
                if d.missing:
                    missing += 1
                else:
                    if not d.aircraft in aircrafts:
                        aircrafts[d.aircraft] = []
                    aircrafts[d.aircraft].append(row)
            #if len(aircrafts)>10:
            #    break

    print 'total aircraft', len(aircrafts), missing
    return aircrafts


if __name__ == '__main__':
    import os
    import sys

    #data_location = "/Users/weihao/Downloads" #'/home/weihao/PY/al'
    data_set = 'training_1_category_4'
    filename = '{0}/{1}/{1}.csv'.format(data_location, data_set)
    truthfile = '{0}/{1}_result/{1}_result.csv'.format(data_location, data_set)
    sensorfile = '{0}/{1}/sensors.csv'.format(data_location, data_set)
    #filename = '{}/A1787.csv'.format(data_location)


    sensors = Sensors(sensorfile)
    truth = TruthData(truthfile)
    aircrafts = read_aircraft(filename)

    #measure = {}
    for a in aircrafts:
        measure = Measurements(aircrafts[a], a, sensors, verbose=(a==2))
        measure.sync()
        # if len(measure)>20:
        #    break

    #data.setup_sensor(sensors.data)

