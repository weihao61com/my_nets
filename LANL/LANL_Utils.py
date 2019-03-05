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
