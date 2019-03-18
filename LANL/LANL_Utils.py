import numpy as np
import os
import pickle
import glob
import random
from scipy import fftpack, fft
from sortedcontainers import SortedDict
import sys

HOME = '/home/weihao/Projects'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects'

sys.path.append('..')
from network import Network

class sNet3(Network):

    def setup(self):
        pass

    def real_setup(self, nodes, outputs, verbose=True):
        self.feed('data')
        for a in range(len(nodes)):
            self.dropout(keep_prob=0.8, name='drop_{}'.format(a))
            self.fc(nodes[a], name= 'fc_{}'.format(a))
            #self.fc_s(nodes[a], name= 'fc_{}'.format(a))

        self.fc(outputs, relu=False, name='output')
        #self.fc_s(outputs, sig=False, name='output')

        if verbose:
            print("number of layers = {} {}".format(len(self.layers), nodes))


class l_utils:

    SEGMENT = 150000

    @staticmethod
    def rdm_ids(files):
        # 0,1,6,11
        # 2,7,12
        # 3,8,13
        # 4,9,14
        # 5,10,15,16
        ix = [-1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, -1]

        ids = SortedDict()
        for f in files:
            strs = f.split('_')
            n = int(strs[-1][:-4])
            ids[n] = f

        idx = {}
        for id in ids:
            idx[ids[id]] = ix[id]

        return idx

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
        return [np.mean(y), l_utils.feature_final(x, dct, dim)]

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
        x = np.array(d)
        #x = x[:,1:]
        #x = x[:,range(0,1000,4)]+x[:,range(1,1000,4)]+x[:,range(2,1000,4)]+x[:,range(3,1000,4)]

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
        avg1, st1, l1 = l_utils.get_c(data[:ll])
        avg2, st2, l2 = l_utils.get_c(data[ll:])
        return map(int, np.array([avg0, st0, avg1, st1, avg2, st2]) * 1000) + [l1, l2]

    @staticmethod
    def get_c(data):
        while True:
            avg = np.mean(data)
            st = np.std(data)*3
            #print len(data), avg, st
            diff = abs(data - avg)
            idx = []
            nt = 0
            for d in diff:
                if d>st:
                    idx.append(nt)
                nt += 1
            if len(idx)==0:
                break
            data = np.delete(data, idx)

        return avg, st, len(data)

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
            win = len(dv)/dim*2
            dv = dv.reshape(dim/2, win)
            d0 = np.std(dv, 1)
            dv = np.mean(dv, 1)
            #ax.plot(dv[:400], c='{}'.format(float(a) / NF), label='{0:5.2f}'.format(np.mean(y)))
            #print '{0:5.2f} {1:5.2f} {2:9.0f}'.format(np.mean(y), d[0].real / len(x), dv[18])
        return np.concatenate((np.array([v0]), dv, d0))

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
    def load_data(filename):
        with open(filename, 'r') as fp:
            lines = fp.readlines()
            if len(lines)==(l_utils.SEGMENT + 1):
                x = []
                for line in lines[1:]:
                    x.append(float(line))
                y = None
            else:
                x = []
                y = []
                for line in lines:
                    a = map(float, line.split(','))
                    x.append(a[0])
                    y.append(a[1])
        return x, y

    @staticmethod
    def get_dataset(subs, st='L_*.p', avg_file='Avg.p'):

        locs = os.path.dirname(subs[0])
        avg_file = os.path.join(locs, avg_file)

        data = []

        for sub in subs:
            sum_file = os.path.join(locs, sub, 'sum.p')
            if not os.path.exists(sum_file):
                #with open(sum_file, 'r') as fp:
                #    sub_data = pickle.load(fp)
                # else:
                sub_data = SortedDict()
                st0 = os.path.join(sub, st)
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
                    # avg0 = np.mean(np.array(v0))
                    with open(avg_file, 'w') as fp:
                        pickle.dump((avg, std), fp)
                else:
                    with open(avg_file, 'r') as fp:
                        A = pickle.load(fp)
                    avg = A[0]
                    std = A[1]
                    # avg0 = A[2]

                for f in d_out:
                    if f not in sub_data:
                        sub_data[f] = []
                    dx = []
                    for d in d_out[f]:
                        dx.append((d[0], (d[1] - avg) / std))
                    sub_data[f] = dx

                with open(sum_file, 'w') as fp:
                    pickle.dump(sub_data,fp)

            data.append(sum_file)


        with open(avg_file, 'r') as fp:
            A = pickle.load(fp)
        avg = A[0]
        att = len(avg)
        return data, att
