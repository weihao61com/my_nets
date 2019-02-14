import csv
import numpy as np


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
            self.measurements[mtx[a, 0]] = list(mtx[a, 1:])

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

    def __init__(self, filename):
        self.header = None
        self.data = {}
        self.test = {}
        with open(filename, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                #print row
                if self.header is None:
                    self.header = row
                    print row
                else:
                    d = Data(row)
                    if d.missing:
                        self.test[d.id] = d
                    else:
                        self.data[d.id] = d
                if len(self.data)>1e6:
                    break
        print 'total data', len(self.data), len(self.test)

    def setup_sensor(self, sensors):
        for s in sensors:
            for d in self.data:
                self.data[d].add_sensor(s)
            for d in self.test:
                self.test[d].add_sensor(s)

class Sensor:
    def __init__(self, rows):
        self.id = int(rows[0])
        self.lla = map(float, rows[1:-1])
        self.name = rows[-1]


class Sensors:
    def __init__(self, filename):
        self.header = None
        self.data = []
        with open(filename, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                if self.header is None:
                    print row
                    self.header = row
                else:
                    self.data.append(Sensor(row))
        print 'Total Sensor', len(self.data)

def validate(data):
    for d in data:
        if len(d.measurements)!=6:
            raise Exception('hah')

if __name__ == '__main__':
    filename = '/home/weihao/PY/al/training_1_category_4/training_1_category_4.csv'
    truthfile = '/home/weihao/PY/al/training_1_category_4_result/training_1_category_4_result.csv'
    sensorfile = '/home/weihao/PY/al/training_1_category_4/sensors.csv'

    sensors = Sensors(sensorfile)
    truth = TruthData(truthfile)
    data = Measurements(filename)

    data.setup_sensor(sensors.data)

    validate(data)