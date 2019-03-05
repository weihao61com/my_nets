import numpy as np


class l_utils:

    @staticmethod
    def get_core(data):

        avg0 = np.mean(data)
        st0 = np.std(data)

        while True:
            avg = np.mean(data)
            st = np.std(data)
            diff = abs(data - avg)
            idx = np.argmax(diff)
            if diff[idx] < st * 3:
                break
            # print len(data), avg
            data = np.delete(data, idx)

        return map(int, np.array([avg0, st0, avg, st]) * 1000)
