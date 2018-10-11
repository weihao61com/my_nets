import sys
import Queue
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/..'.format(this_file_path))
from utils import Utils
from o2_load import *
from network import Network


def _reshuffle(data):
    np.random.shuffle(data[0])


def _reshuffle_b(bucket):
    for data in bucket:
        np.random.shuffle(data[0])


class DataSet:
    def __init__(self, dataset, batch_size=500, npar=50, cache=True):
        self.bucket = 0
        self.dataset = dataset
        self.index = -1
        self.batch_size = batch_size
        self.nPar = npar
        self.data = None
        self.verbose = True
        self.cache = cache
        self.memories = {}

        self.load_next_data()
        self.sz = self.data[0][0][0].shape
        self.id = None

    def get_next(self, rd=False, num_output=3):
        self.load_next_data()

        if self.index == 0:
            return None
        return self.prepare(rd, num_output=num_output)

    def load_next_data(self):
        self.bucket = 0

        if len(self.dataset)==1 and self.index==0:
            return

        self.index += 1
        if self.index == len(self.dataset):
            self.verbose = False
            self.index = 0
        if self.index in self.memories:
            data = self.memories[self.index]
        else:
            data = load_data(self.dataset[self.index], self.verbose)

        if self.cache:
            self.memories[self.index] = data

        self.data = []
        for a in range(0, len(data), self.batch_size):
            b = a + self.batch_size
            if b>len(data):
                b=len(data)
            self.data.append(data[a:b])

    def prepare_cnn(self, rd=False, num_output=3):
        pre_data = []
        self.reshuffle_data()
        for d in self.data:
            pre_data.append(self.create_bucket_cnn(d, num_output))
        if rd:
            np.random.shuffle(pre_data)
        return pre_data

    def prepare(self, rd=True, num_output=3):
        pre_data = []
        self.reshuffle_data()
        self.id = 0
        for d in self.data:
            pre_data.append(self.create_bucket(d, num_output))
        if rd:
            np.random.shuffle(pre_data)
        return pre_data

    def create_bucket(self, data, num_output):
        outputs = []
        inputs = []
        ids = []
        sz_in = data[0][0].shape

        for d in data:
            input = d[0]
            num = 5 #int(np.ceil(len(input)/float(self.nPar)))
            length = num*self.nPar
            while len(input) < length: #self.nPar:
                input= np.concatenate((input, input))
            input = input[:length]
            for a in range(0, len(input), self.nPar):
                it = input[a:a+self.nPar]
                truth = d[1][:num_output]
                inputs.append(it.reshape(self.nPar * sz_in[1]))
                outputs.append(truth.reshape(num_output))
                ids.append(self.id)
            self.id += 1

        return np.array(inputs), np.array(outputs), ids

    def create_bucket_cnn(self, data, num_output):
        outputs = []
        inputs = []
        sz_in = data[0][0].shape

        for d in data:
            input = d[0]
            while len(input) < self.nPar:
                input= np.concatenate((input, input))
            input = input[:self.nPar]
            truth = d[1][:num_output]
            inputs.append(input.reshape((self.nPar, sz_in[1], 1)))
            outputs.append(truth.reshape(num_output))

        return inputs, outputs

    def reshuffle_data(self):
        from multiprocessing.pool import ThreadPool
        import multiprocessing

        #pool = ThreadPool(multiprocessing.cpu_count() - 2)
        #pool.map(_reshuffle_b, self.data)
        #pool.close()
        #pool.join()

        for bucket in self.data:
            _reshuffle_b(bucket)

    def prepare_slow(self, num_output=3):
        pre_data = []
        for a in range(len(self.data)):
            data_gen = self.gen_data(self.nPar, num_output)
            sz_in = self.data[0][0][0].shape
            inputs = []
            outputs = []
            self.bucket = a
            while True:
                input_p, output_p = next(data_gen, (None, None))
                if input_p is None:
                    break
                inputs.append(input_p.reshape(self.nPar*sz_in[1]))
                outputs.append(output_p.reshape(num_output))

            pre_data.append((inputs, outputs))

        self.bucket = len(pre_data)
        return pre_data

    def gen_data(self, nPar, num_output):
        #np.random.seed()
        indices = range(len(self.data[self.bucket]))

        for a in indices:
            input0 = []
            while nPar>len(input0):
                input = self.data[self.bucket][a][0]
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

            output = self.data[self.bucket][a][1][:num_output]
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

class sNet3_stage(Network):

    def setup(self):
        nodes = [1024, 64]
        (self.feed('data').
         fc(nodes[0], name='fc1_stage').
         fc(nodes[1], name='fc2_stage').
         fc(3, relu=False, name='output_stage'))

        print("number of layers = {} {}".format(len(self.layers), nodes))

class sNet3(Network):

    def setup(self):
        nodes = [2048,256]
        (self.feed('data').
         fc(nodes[0], name='fc2').
         fc(nodes[1], name='fc3').
         fc(3, relu=False, name='output'))

        print("number of layers = {} {}".format(len(self.layers), nodes))


class cNet(Network):

    def setup(self):
        nodes = [16, 1024, 32]
        (self.feed('data')
         .conv(1, 4, nodes[0], 1, 1, name='conv1', padding='VALID')
         .fc(nodes[1], name='fc1')
         .fc(nodes[2], name='fc2')
         .fc(3, relu=False, name='output'))

        print("number of layers = {} {}".
              format(len(self.layers), nodes))



def run_data(data, inputs, sess, xy):
    results = None
    truth = None

    for b in data:
        feed = {inputs: b[0]}
        result = sess.run(xy, feed_dict=feed)
        if results is None:
            results = result
            truth = b[1]
        else:
            results = np.concatenate((results, result))
            truth = np.concatenate((truth, b[1]))

    return Utils.calculate_loss(results, truth)


def run_stage_data(data, inputs, sess, xy, stage_input, xy_stage):
    results = None
    truth = None
    num_att = 4
    stage_dup = 5
    stage_result = None

    for b in data:
        feed = {inputs: b[0]}
        result = sess.run(xy, feed_dict=feed)
        if results is None:
            results = result
            truth = b[1]
        else:
            results = np.concatenate((results, result))
            truth = np.concatenate((truth, b[1]))

        length = b[0].shape[1]
        for evt in range(0, length, num_att*stage_dup):
            input_array = np.concatenate((result, b[0][:, evt:evt + num_att*stage_dup]), axis=1)
            r = sess.run(xy_stage, feed_dict={stage_input: input_array}) + result - b[1]
            if stage_result is None:
                stage_result = np.linalg.norm(r, axis=1)
            else:
                stage_result = np.concatenate((stage_result, np.linalg.norm(r, axis=1)))

    a, b = Utils.calculate_loss(results, truth)
    avg = stage_result.dot(stage_result)/len(stage_result)
    return a, b, avg
