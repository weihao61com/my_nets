import numpy as np
import os
import pickle
import glob
import random
from scipy import fftpack, fft
from sortedcontainers import SortedDict
import sys

sys.path.append('..')
from network import Network

class sNet3(Network):

    def setup(self):
        pass

    def real_setup(self, nodes, outputs):
        self.feed('data')
        for a in range(len(nodes)):
            self.dropout(keep_prob=0.3, name='drop_{}'.format(a))
            self.fc(nodes[a], name= 'fc_{}'.format(a))
            #self.fc_s(nodes[a], name= 'fc_{}'.format(a))

        self.fc(outputs, relu=False, name='output')
        #self.fc_s(outputs, sig=False, name='output')

        print("number of layers = {} {}".format(len(self.layers), nodes))


class l_utils:

    @staticmethod
    def get_features(lines, dct, dim):
        #if not len(lines) == SEG:
        #    raise Exception("Wrong data length {}".format(len(lines)))

        x = []
        y = []
        for line in lines:
            v = map(float, line.split(','))
            x.append(v[0])
            y.append(v[1])

        # return np.mean(y), l_utils.fft_feature_final(x, win, dolog)
        return np.mean(y), l_utils.feature_final(x, dct, dim)

    @staticmethod
    def prepare_data(sum_file, c, rd=False):
        data = []
        with open(sum_file, 'r') as fp:
            sub_data = pickle.load(fp)
        nt = 1
        for dd in sub_data:
            if c>0 and nt==c:
                data = data + sub_data[dd]
            if c<0 and nt!=-c:
                data = data + sub_data[dd]
            nt += 1

        if rd:
            random.shuffle(data)

        t = []
        d = []

        for dd in data:
            t.append(dd[0])
            d.append(dd[1])
        d = np.array(d)
        d = d[:,1:]
        x = d[:,range(0,1000,4)]+d[:,range(1,1000,4)]+d[:,range(2,1000,4)]+d[:,range(3,1000,4)]

        return np.array(t), x

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
    def fft_feature_final(x, win = 100, dolog=False):
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
        return np.concatenate((np.array([v0]), avg))

    @staticmethod
    def feature_final(x, dct, dim):
        if dct:
            d = fftpack.dct(x)
            win = len(d) / dim
            v0 = d[0]
            d[0] = d[1]
            dv = np.array(d).reshape(dim, win)
            dv = np.std(dv, 1)
            #ax.plot(dv[:400], c='{}'.format(float(a) / NF), label='{0:5.2f}'.format(np.mean(y)))
            #print '{0:5.2f} {1:5.2f} {2:5.2f}'.format(np.mean(y), d[0] / len(x), dv[18])
        else:
            d = abs(fft(x))
            v0 = d[0]
            dv = np.array(d[1:len(d)/2+1])
            win = len(dv)/dim
            dv = dv.reshape(dim, win)
            # dv = np.std(dv, 1)
            dv = np.mean(dv, 1)
            #ax.plot(dv[:400], c='{}'.format(float(a) / NF), label='{0:5.2f}'.format(np.mean(y)))
            #print '{0:5.2f} {1:5.2f} {2:9.0f}'.format(np.mean(y), d[0].real / len(x), dv[18])
        return np.concatenate((np.array([v0]), dv))

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
    def get_dataset(locs, subs, st='L_*.p', avg_file='Avg.p'):

        avg_file = os.path.join(locs, avg_file)
        data = []

        for sub in subs:
            sum_file = os.path.join(locs, sub, 'sum.p')
            if not os.path.exists(sum_file):
                #with open(sum_file, 'r') as fp:
                #    sub_data = pickle.load(fp)
                # else:
                sub_data = SortedDict()
                st0 = os.path.join(locs, sub, st)
                d_out = {}

                files = glob.glob(st0)
                for f in files:
                    basename = os.path.basename(f)
                    if basename not in d_out:
                        d_out[basename] = []
                    with open(f, 'r') as fp:
                        d = pickle.load(fp)
                        print 'File', f, len(d), len(d[0][1])
                        d_out[basename] = d

                if not os.path.exists(avg_file):
                    v0 = []
                    v1 = []
                    for f in d_out:
                        for d in d_out[f]:
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

                for f in d_out:
                    if f not in sub_data:
                        sub_data[f] = []
                    dx = []
                    for d in d_out[f]:
                        dx.append((d[0] - avg0, (d[1] - avg) / std))
                    sub_data[f] = dx

                with open(sum_file, 'w') as fp:
                    pickle.dump(sub_data,fp)

            data.append(sum_file)

        return data
