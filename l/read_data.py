import csv
import numpy as np
from sortedcontainers import SortedDict
from sklearn import linear_model


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


class Measurements:

    def __init__(self, rows, a, sensors, verbose=True):
        self.data = SortedDict()
        self.sensors = sensors

        for row in rows:
            d = Data(row)
            self.data[d.id] = d

        print 'total data', a, len(self.data)

        if verbose:
            filename = '/Users/weihao/tmp/m_{}.csv'.format(a)
            with open(filename, 'w') as fp:
                for t in self.data:
                    dd = self.data[t]
                    for m in dd.measurements:
                        mm = dd.measurements[m]
                        ss = sensors.data[m]
                        lla = ss.lla
                        fp.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.
                                 format(dd.timeAtServer, dd.id, dd.lat, dd.lon, dd.bAlt, dd.gAlt,
                                        m, mm[0], mm[1], lla[0], lla[1], lla[2]))

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
                print m, len(sensor_list[m]), z[0], z[1]

    def get_train_data(self, K_measure):
        train = []
        ids = self.data.keys()

        for a in range(len(ids)):
            id = ids[a]
            td = TrainData(id, K_measure)
            a1 = a
            a2 = a
            dd = self.data[id]
            T0 = dd.timeAtServer
            if not dd.missing:
                td.add_data(dd.measurements)
            while not td.enough():
                a1 -= 1
                a2 += 1
                if a1<0:
                    aa = a2
                elif a2>=len(ids):
                    aa = a1
                else:
                    d1 = self.data[ids[a1]]
                    d2 = self.data[ids[a2]]
                    if T0-d1.timeAtServer > d2.timeAtServer-T0:
                        aa = a2
                    else:
                        aa = a1
                id = ids[aa]
                dd = self.data[id]
                if not dd.missing:
                    td.add_data(dd.measurements)
            train.append(td)

        return train


class Sensor:
    def __init__(self, rows):
        self.id = int(rows[0])
        self.lla = map(float, rows[1:-1])
        self.name = rows[-1]
        self.s_id = None


class Sensors:
    def __init__(self, filename):
        self.header = None
        self.data = {}
        names = {}
        sensor_id = 0
        with open(filename, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                if self.header is None:
                    print row
                    self.header = row
                else:
                    s = Sensor(row)
                    self.data[s.id] = s
                    if not s.name in names:
                        names[s.name] = sensor_id
                        sensor_id += 1

        print 'Total Sensor', len(self.data), len(names)
        print names
        for id in self.data:
            self.data[id].s_id = names[self.data[id].name]


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

    K_measure = 50
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    data_location = '{}/../../datasets/al'.format(this_file_path)

    #data_location = "/Users/weihao/Downloads" #'/home/weihao/PY/al'
    data_set = 'training_1_category_4'
    filename = '{0}/{1}/{1}.csv'.format(data_location, data_set)
    truthfile = '{0}/{1}_result/{1}_result.csv'.format(data_location, data_set)
    sensorfile = '{0}/{1}/sensors.csv'.format(data_location, data_set)
    filename = '{}/A1787.csv'.format(data_location)


    sensors = Sensors(sensorfile)
    truth = TruthData(truthfile)
    aircrafts = read_aircraft(filename)

    measure = {}
    for a in aircrafts:
        measure[a] = Measurements(aircrafts[a], a, sensors, verbose=True)
        measure[a].sync()
        # if len(measure)>20:
        #    break

    data = []
    for a in measure:
        train = measure[a].get_train_data(K_measure)

    #data.setup_sensor(sensors.data)

