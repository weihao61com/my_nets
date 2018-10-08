import numpy as np

class Utils_SN:

    @staticmethod
    def read_ref(filename):
        with open(filename, 'r') as fp:
            line = fp.readline()
            vals = np.array(map(float, line.split()))
        return vals.reshape((1,len(vals)))
