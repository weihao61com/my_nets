import pickle
import numpy as np
import random


def increment(len, idx):
    idx += 1
    if idx == len:
        idx = 0
    return idx


class o2_event:

    def __init__(self, data, stack=2):
        self.data = data[0][:20]
        self.truth = data[1]
        self.len = len(self.data)/stack
        self.data_len = len(self.data)
        self.stack = stack

    def data_gen(self, reshuffle=True):
        indices = range(self.data_len)
        if reshuffle:
            random.shuffle(indices)
        idx = -1
        while True:
            out = []
            for a in range(self.stack):
                idx = increment(self.data_len, idx)
                a1 = self.data[indices[idx]].reshape((1, 2))
                out.append(a1)
            yield out, self.truth[:4].reshape((1, 4))
#
#
# class o2_data:
#
#     def __init__(self, location, filename):
#         self.data = load_data(location + '/' + filename)[:50]
#         self.len = len(self.data)
#
#     def event_gen(self, stack, reshuff=True):
#         indices = range(self.len)
#         if reshuff:
#             random.shuffle(indices)
#         idx = -1
#         while True:
#             idx = increment(self.len, idx)
#             #print 'event_gen', idx, indices[idx]
#             yield o2_event(self.data[indices[idx]], stack)


def load_data(filename, verbose = True, max_len=2000):
    import datetime
    t0 = datetime.datetime.now()
    if type(filename) is unicode:
        avg_param = 0
        with open(filename, 'r') as fp:
            data = pickle.load(fp)
            for d in data:
                avg_param += len(d[0])
            length = len(data)
            avg_param /= length
            if length>max_len:
                data = data[:max_len]
        if verbose:
            print "loading", filename, datetime.datetime.now()-t0, \
                length, len(data), avg_param
        return data
    data_out = None
    for f in filename:
        data = load_data(f)
        if data_out is None:
            data_out = data
        else:
            data_out = np.concatenate((data_out, data))

    lenght = len(data_out)
    if lenght > max_len:
        data_out = data_out[:max_len]
    print "Total data", datetime.datetime.now() - t0, lenght, len(data_out)
    return data_out

def gen_data(source):
    while True:
        indices = range(len(source))
        np.random.shuffle(indices)
        for a in indices:
            input = source[a][0]
            output = source[a][1][:2]
            yield input, output


def gen_data_batch(source, batch_size):
    data_gen = gen_data(source)
    while True:
        inputs = []
        outputs = []
        for _ in range(batch_size):
            input, output = next(data_gen)
            sz_in = input.shape
            np.random.shuffle(input)
            inputs.append(input.reshape(sz_in[0] * sz_in[1]))
            outputs.append(output.reshape(2))

        yield np.array(inputs), np.array(outputs)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = load_data('data.p')
    print len(data)
    print len(data[0])
    print data[0][0].shape, data[0][1].shape
    print data[1][0].shape, data[1][1].shape
    print data[2][0].shape, data[2][1].shape

    print data[0][1]
    plt.plot(data[0][0][:, 0], data[0][0][:, 1], '.')
    plt.show()