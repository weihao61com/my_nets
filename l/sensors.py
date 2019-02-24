import csv
import numpy as np
from sortedcontainers import SortedDict
from sklearn import linear_model
import os
# import utm
import pyproj
import math
from L_utils import *
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
    filename = 'A103.csv'.format(data_location)

    sensors = Sensors(sensorfile)
    aircrafts, missing = LUtils.read_aircraft(filename)

    s_id = 152
    sensor_reading = {}
    for id in aircrafts:
        measure = Measurements(aircrafts[id], id, sensors, fp=None)
        for id in measure.data:
            dd = measure.data[id]
            for m in dd.measurements:
                mm = dd.measurements[m]
                ss = sensors.data[m]
                dist = np.linalg.norm(ss.xyz-dd.xyz)
                if ss.id==s_id:
                    dt = dd.timeAtServer- mm[0]*1e-9
                    print 'sn', id, ss.id,ss.type_id,mm[1],dd.timeAtServer, mm[0]*1e-9, dt, dist/SPEED