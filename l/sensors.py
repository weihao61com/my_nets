import csv
import numpy as np
from sortedcontainers import SortedDict
from sklearn import linear_model
import os
# import utm
import pyproj
import math
from L_utils import LUtils, Sensors, Measurements
import pickle

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_location = '{}/../../datasets/al'.format(this_file_path)


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
    aircrafts = LUtils.read_aircraft(filename)

    for id in aircrafts:
        sensor_reading = {}
        measure = Measurements(aircrafts[id], id, sensors, verbose=False)
        #measure.sync()
        for t in measure.data:
            dd = measure.data[t]
            for m in dd.measurements:
                mm = dd.measurements[m]
                ss = sensors.data[m]
                if ss.id==322:
                    dt = dd.timeAtServer-mm[0]*1e-9
                    print 'sn', ss.id,ss.type_id,mm[1],dd.timeAtServer,dd.lat,dd.lon,dd.bAlt,id, mm[0]*1e-9, dt