import numpy as np
import datetime
import pickle
from L_utils import LUtils


def load_go(filename='global_offset.csv'):
    return np.array(LUtils.read_csv(filename)).astype(float)

class DataSet:
    def __init__(self, dataset, cfg, cache=True, sub_sample=-1):
        self.bucket = 0
        self.dataset = dataset
        self.index = -1
        self.data = None
        self.verbose = True
        self.cache = cache
        self.memories = {}
        self.num_output = 3
        self.go = load_go()

        if cfg is not None:
            self.t_scale = cfg.t_scale
            self.net_type = cfg.net_type
            self.att = cfg.att
            self.batch_size = cfg.memory_size
            self.nPar = cfg.feature_len
            self.nAdd = cfg.add_len
            self.num_output = cfg.num_output

        self.load_next_data(sub_sample)
        self.sz = self.data[0][0][0].shape
        self.id = None
        #self.att = self.sz[1]

    def load_data(self, filename):
        t0 = datetime.datetime.now()
        if type(filename) is unicode or type(filename) is str:
            data0 = []
            avg_param = 0
            nr = 0
            nt = 0
            with open(filename, 'r') as fp:
                data = pickle.load(fp)
                for flight_id in data:
                    flight = data[flight_id]
                    for id in flight:
                        truth = self.create_truth(flight[id][1])
                        for r in self.create_data(flight[id][0]):
                            data0.append((r, truth))
            return data0
        data_out = None
        for f in filename:
            data = self.load_data(f)
            if data_out is None:
                data_out = data
            else:
                data_out = np.concatenate((data_out, data))

        lenght = len(data_out)

        print "Total data", datetime.datetime.now() - t0, lenght, len(data_out)
        return data_out

    def create_data(self, data):
        d = []
        D0 = None
        for a in data[1]:
            if D0 is None:
                D0 = [a] + data[1][a]
            else:
                D1 = [a] + data[1][a]
                d.append([data[0], D0[0], D0[2], D1[0], D1[2], (D0[1]-D1[1])*1e-6])
        return d

    def create_truth(self, data):
        return (np.array(data) - self.go[:, 0])/self.go[:, 1]

    def get_next(self, rd=True):
        self.load_next_data()

        if self.index == 0:
            return None
        return self.prepare(rd)

    def load_next_data(self, sub_sample=-1):
        self.bucket = 0

        if len(self.dataset) == 1 and self.index == 0:
            return

        self.index += 1
        if self.index == len(self.dataset):
            self.verbose = False
            self.index = 0
        if self.index in self.memories:
            data = self.memories[self.index]
        else:
            data = self.load_data(self.dataset[self.index])
            att = data[0][0].shape[1]
            if self.att<att:
                for a in range(len(data)):
                    data[a][0] = data[a][0][:,:self.att]

        if self.cache:
            self.memories[self.index] = data

        self.data = []
        step = np.ceil(float(len(data))/self.batch_size)
        step = int(len(data)/step)
        print 'Step', step, len(data)
        for a in range(0, len(data), step):
            b = a + step
            if b > len(data):
                b = len(data)
            self.data.append(data[a:b])

    def prepare_cnn(self, rd=False):
        pre_data = []
        self.reshuffle_data()
        for d in self.data:
            pre_data.append(self.create_bucket_cnn(d))
        if rd:
            np.random.shuffle(pre_data)
        return pre_data

    def prepare_stack(self, rd=True):
        pre_data = []
        self.reshuffle_data()
        self.id = 0
        for d in self.data:
            data = self.create_stack(d)
            if rd:
                np.random.shuffle(data)
            pre_data.append(data)

        return pre_data

    def create_stack(self, data):
        outputs = []

        for d in data:
            sz = d[0].shape
            if sz[0] < self.nPar + self.nAdd:
                d[0] = np.concatenate((d[0], d[0]))
                sz = d[0].shape
            truth = d[1][:self.num_output]
            d0 = d[0][:self.nPar].reshape(self.nPar * sz[1])
            d1 = d[0][self.nPar:]
            outputs.append([d0, d1, truth])

        return outputs

    def prepare_stage(self, rd=True, rdd=True):
        pre_data = []
        if rdd:
            self.reshuffle_data()
        self.id = 0
        for d in self.data:
            data = self.create_stage_data(d)
            if rd:
                np.random.shuffle(data)
            i0 = []
            i1 = []
            outs = []
            ids = []
            for a in data:
                i0.append(a[0])
                i1.append(a[1])
                outs.append(a[2])
                ids.append(a[3])
            dd = (np.array(i0), np.array(i1), np.array(outs), ids)
            pre_data.append(dd)

        return pre_data

    def prepare(self, rd=True, multi=1, rdd=True):
        pre_data = []
        if rdd:
            self.reshuffle_data()
        self.id = 0
        for d in self.data:
            data = self.create_bucket(d, multi)
            if rd:
                np.random.shuffle(data)
            ins = []
            outs = []
            ids = []
            for a in data:
                if self.net_type == 'fc':
                    ins.append(a[0])
                elif self.net_type == 'cnn':
                    ins.append(a[0].reshape(((self.nPar + self.nAdd), self.att,1)))
                else:
                    raise Exception()
                outs.append(a[1]*self.t_scale)
                ids.append(a[2])
            dd = (np.array(ins), np.array(outs), ids)
            pre_data.append(dd)

        return pre_data

    def create_stage_data(self, data):
        multi = 10
        f2 = 1
        N1 = self.nPar
        N2 = int(self.nPar*f2)
        length = multi*(N1+N2)
        outputs = []
        for d in data:
            input = d[0]
            while len(input) < length:
                input = np.concatenate((input, input))

            input = input[:length]
            for m in range(multi):
                start = m*(N1+N2)
                i1 = input[start:start+N1].reshape(N1 * self.sz[1])
                i2 = input[start+N1:start+N1+N2]
                truth = d[1][:self.num_output]
                output = (i1, i2,  truth.reshape(self.num_output), self.id)
                outputs.append(output)

        self.id += 1

        return outputs


    def create_bucket(self, data, multi):

        outputs = []

        sz_in = data[0][0].shape

        for d in data:
            input = d[0]
            if multi > 0:
                num = multi  # *int(np.ceil(len(input)/float(self.nPar)))
            else:
                num = int(np.ceil(len(input) / float(self.nPar + self.nAdd))*abs(multi))
            length = num * (self.nPar + self.nAdd)
            while len(input) < length:
                input = np.concatenate((input, input))
            input = input[:length]
            for a in range(0, len(input), self.nPar+self.nAdd):
                it = input[a:a + self.nPar+self.nAdd]
                truth = d[1][:self.num_output]
                output = (it.reshape((self.nPar+self.nAdd) * sz_in[1]),
                          truth.reshape(self.num_output), self.id)
                outputs.append(output)

            self.id += 1

        return outputs

    def create_bucket_cnn(self, data):
        outputs = []
        inputs = []
        sz_in = data[0][0].shape

        for d in data:
            input = d[0]
            while len(input) < self.nPar:
                input = np.concatenate((input, input))
            input = input[:self.nPar]
            truth = d[1][:self.num_output]
            inputs.append(input.reshape((self.nPar, sz_in[1], 1)))
            outputs.append(truth.reshape(self.num_output))

        return inputs, outputs

    def reshuffle_data(self):
        from multiprocessing.pool import ThreadPool
        import multiprocessing

        # pool = ThreadPool(multiprocessing.cpu_count() - 2)
        # pool.map(_reshuffle_b, self.data)
        # pool.close()
        # pool.join()

        for bucket in self.data:
            _reshuffle_b(bucket)

    def prepare_slow(self):
        pre_data = []
        for a in range(len(self.data)):
            data_gen = self.gen_data(self.nPar)
            sz_in = self.data[0][0][0].shape
            inputs = []
            outputs = []
            self.bucket = a
            while True:
                input_p, output_p = next(data_gen, (None, None))
                if input_p is None:
                    break
                inputs.append(input_p.reshape(self.nPar * sz_in[1]))
                outputs.append(output_p.reshape(self.num_output))

            pre_data.append((inputs, outputs))

        self.bucket = len(pre_data)
        return pre_data

    def gen_data(self, nPar):
        # np.random.seed()
        indices = range(len(self.data[self.bucket]))

        for a in indices:
            input0 = []
            while nPar > len(input0):
                input = self.data[self.bucket][a][0]
                np.random.shuffle(input)
                if len(input0) == 0:
                    if nPar < len(input):
                        input0 = input[:nPar]
                    else:
                        input0 = input
                else:
                    if len(input) + len(input0) > nPar:
                        dl = nPar - len(input0)
                        input0 = np.concatenate((input0, input[:dl, :]))
                    else:
                        input0 = np.concatenate((input0, input))

            output = self.data[self.bucket][a][1][:self.num_output]
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


if __name__ == '__main__':
    filename = '/Users/weihao/Projects/p_files/training_1_category_4_0.p'
    ds = DataSet([filename], None)