import numpy as np
import os
import pickle
import glob
import random

class l_utils:

    @staticmethod
    def prepare_data(input, rd=False):
        data = []
        for dd in input:
            data = data + dd

        if rd:
            random.shuffle(data)

        t = []
        d = []

        for dd in data:
            t.append(dd[0])
            d.append(dd[1])

        return np.array(t), np.array(d)

    @staticmethod
    def prepare_rf_data(input):
        data = []
        for dd in input:
            data = data + dd

        t = []
        for dd in data:
            t.append(dd[0])
        return data, t

    @staticmethod
    def get_core(data):

        avg0 = np.mean(data)
        st0 = np.std(data)

        ll = int(len(data)/2)
        avg1, st1 = l_utils.get_c(data[:ll])
        avg2, st2 = l_utils.get_c(data[ll:])
        return map(int, np.array([avg0, st0, avg1, st1, avg2, st2]) * 1000)

    @staticmethod
    def get_c(data):
        while True:
            avg = np.mean(data)
            st = np.std(data)
            diff = abs(data - avg)
            idx = np.argmax(diff)
            if diff[idx] < st * 3:
                break
            # print len(data), avg
            data = np.delete(data, idx)

        return avg, st

    @staticmethod
    def csv_line(dd):
        output = None
        for d in dd:
            if output is None:
                output = '{}'.format(d)
            else:
                output = '{},{}'.format(output, d)

        return output

    @staticmethod
    def fft_features(x, win = 1000):
        x = np.array(x)
        length = len(x)
        f = np.fft.fft(x)
        f = abs(f)

        v0 = f[0]

        f = f[1:length / 2 + 1]
        dim = len(f)/win
        af = f.reshape(dim, win)
        avg = np.mean(af, 1)
        std = np.std(af, 1)
        return v0, avg, std

    @staticmethod
    def fft_feature_final(x, win = 100, rg=200, dolog=False):
        x = np.array(x)
        length = len(x)
        f = np.fft.fft(x)
        f = abs(f)

        v0 = f[0]

        f = f[1:length / 2 + 1]
        dim = len(f)/win
        af = f.reshape(dim, win)
        avg = np.mean(af, 1)

        if dolog:
            v0 = np.log(v0)
            avg = np.log(avg)

        #std = np.std(af, 1)
        return np.concatenate((np.array([v0]), avg[:rg]))

    @staticmethod
    def csv_line(dd):
        output = None
        for d in dd:
            if output is None:
                output = '{}'.format(d)
            else:
                output = '{},{}'.format(output, d)

        return output

    @staticmethod
    def get_dataset(locs='/home/weihao/Projects/p_files', st='L_*.p', avg_file='Avg.p'):

        st = os.path.join(locs, st)
        avg_file = os.path.join(locs, avg_file)

        files = glob.glob(st)
        data = []
        for f in files:
            with open(f, 'r') as fp:
                d = pickle.load(fp)
                data.append(d)

        if not os.path.exists(avg_file):
            v0 = []
            v1 = []
            for dd in data:
                for d in dd:
                    v0.append(d[0])
                    v1.append(d[1])
            v1 = np.array(v1)
            avg = np.mean(v1, 0)
            std = np.std(v1, 0)
            avg0 = np.mean(np.array(v0))
            with open(avg_file, 'w') as fp:
                pickle.dump((avg, std, avg0), fp)
        else:
            with open(avg_file, 'r') as fp:
                A = pickle.load(fp)
            avg = A[0]
            std = A[1]
            avg0 = A[2]

        d_out = []
        for dd in data:
            dx = []
            for d in dd:
                dx.append((d[0] - avg0, (d[1] - avg) / std))
            d_out.append(dx)
        return d_out
