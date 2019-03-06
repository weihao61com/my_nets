import numpy as np


class l_utils:

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
    def csv_line(dd):
        output = None
        for d in dd:
            if output is None:
                output = '{}'.format(d)
            else:
                output = '{},{}'.format(output, d)

        return output