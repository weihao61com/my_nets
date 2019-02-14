import csv
import numpy as np
from sortedcontainers import SortedDict


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


class Measurements:

    def __init__(self, rows, a, sensors, verbose=True):
        self.data = SortedDict()

        for row in rows:
            d = Data(row)
            self.data[d.timeAtServer] = d

        print 'total data', a, len(self.data)

        if verbose:
            filename = '/home/weihao/tmp/m_{}.csv'.format(a)
            with open(filename, 'w') as fp:
                for t in self.data:
                    dd = self.data[t]
                    for m in dd.measurements:
                        mm = dd.measurements[m]
                        ss = sensors.data[m]
                        lla = ss.lla
                        fp.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.
                                 format(t, dd.id, dd.lat, dd.lon, dd.bAlt, dd.gAlt,
                                        m, mm[0], mm[1], lla[0], lla[1], lla[2]))


class Sensor:
    def __init__(self, rows):
        self.id = int(rows[0])
        self.lla = map(float, rows[1:-1])
        self.name = rows[-1]


class Sensors:
    def __init__(self, filename):
        self.header = None
        self.data = {}
        with open(filename, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                if self.header is None:
                    print row
                    self.header = row
                else:
                    s = Sensor(row)
                    self.data[s.id] = s
        print 'Total Sensor', len(self.data)


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
    data_location = '/home/weihao/PY/al'
    data_set = 'training_1_category_4'
    filename = '{0}/{1}/{1}.csv'.format(data_location, data_set)
    truthfile = '{0}/{1}_result/{1}_result.csv'.format(data_location, data_set)
    sensorfile = '{0}/{1}/sensors.csv'.format(data_location, data_set)

    sensors = Sensors(sensorfile)
    truth = TruthData(truthfile)
    aircrafts = read_aircraft(filename)

    measure = {}
    for a in aircrafts:
        measure[a] = Measurements(aircrafts[a], a, sensors)
        if len(measure)>4:
            break

    #data.setup_sensor(sensors.data)

