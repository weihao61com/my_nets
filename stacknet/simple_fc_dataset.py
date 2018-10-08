# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
import cv2
import datetime
import sys
import Queue
import copy

#HOME = '/home/weihao/Projects/Posenet/'
sys.path.append('..')
from o2_load import *
from network import Network
import time

class DataSet:
    def __init__(self, data, batch_size=500):
        self.data = data
        self.batch_size = batch_size
        self.pre_data = None
        self.length = len(data)
        self.sz = data[0][0].shape
        self.ready = False
        self.bucket = 0

    def prepare(self, rd=False, nPar=50, num_output=2):
        data_gen = self.gen_data(rd, nPar, num_output)
        sz_in = self.data[0][0].shape
        pre_data = []

        while True:
            inputs = []
            outputs = []
            done = False
            for _ in range(self.batch_size):
                input_p, output_p = next(data_gen, (None, _))
                if input_p is None:
                    done = True
                    break
                # output_p = output_p*np.pi/180.0
                inputs.append(input_p.reshape(nPar*sz_in[1]))
                outputs.append(output_p.reshape(num_output))

            if len(inputs) > 0:
                pre_data.append((inputs, outputs))
            if done:
                break
        # print 'bucket', len(self.pre_data)
        self.bucket = len(pre_data)
        return pre_data

    def gen_data(self, rd, nPar, num_output):
        #np.random.seed()
        indices = range(len(self.data))
        if rd:
            np.random.shuffle(indices)

        for a in indices:
            input0 = []
            while nPar>len(input0):
                input = copy.copy(self.data[a][0])
                np.random.shuffle(input)
                if len(input0)==0:
                    if nPar<len(input):
                        input0 = input[:nPar]
                    else:
                        input0 = input
                else:
                    if len(input)+len(input0)>nPar:
                        dl = nPar - len(input0)
                        input0 = np.concatenate((input0, input[:dl, :]))
                    else:
                        input0 = np.concatenate((input0, input))

            output = self.data[a][1][:num_output]
            yield input0, output

    def q_fun(self, id, rst_dict):
        rst_dict[id] = self.prepare()

    def prepare2(self, rd=True):
        data_gen = self.gen_data(rd)
        sz_in = self.data[0][0].shape
        pre_data = []

        while True:
            inputs = []
            outputs = []
            done = False
            for _ in range(self.batch_size):
                input_p, output_p = next(data_gen, (None, _))
                if input_p is None:
                    done = True
                    break
                inputs.append(input_p.reshape(sz_in[0], sz_in[1], 1))
                outputs.append(output_p.reshape(2))

            if len(inputs) > 0:
                pre_data.append((inputs, outputs))
            if done:
                break
        # print 'bucket', len(self.pre_data)
        self.bucket = len(pre_data)
        return pre_data


def get_queue_not(tr, pool):
    if pool == 1:
        queue = Queue.Queue()
        queue.put(tr.prepare())
        return queue

    import multiprocessing

    manager = multiprocessing.Manager()
    trs = []
    worker_count = pool
    rst_dict = manager.dict()

    for w in range(worker_count):
        p = multiprocessing.Process(target=tr.q_fun,
                                    args=(w, rst_dict))
        p.start()
        trs.append(p)

    for p in trs:
        p.join()

    queue = Queue.Queue()
    for w in range(worker_count):
        queue.put(rst_dict[w])

    return queue

def cal_diff(t, r):
    r = np.array(r)
    mm = np.median(r, 0)
    #ss = np.std(r, 1)
    #d0 = t[0]-mm[0]
    #d1 = t[1]-mm[1]
    dd = np.linalg.norm(t - mm)
    dd = dd*dd
    # dd = dd*dd

    return dd, mm

class sNet1(Network):

    def setup(self):
        (self.feed('data').
         fc(2048, name='fc1').
         fc(128, name='fc2').
         fc(1, relu=False, name='output'))

        print("number of layers = {}".format(len(self.layers)))

class sNet(Network):

    def setup(self):
        (self.feed('data').
         fc(2048, name='fc1').
         fc(128, name='fc2').
         fc(3, relu=False, name='output'))

        print("number of layers = {}".format(len(self.layers)))


class cNet(Network):

    def setup(self):
        (self.feed('data')
         .conv(1, 2, 16, 1, 1, name='conv1')
         .fc(40, name='fc1')
         .fc(2, relu=False, name='output'))

        print("number of layers = {}".format(len(self.layers)))