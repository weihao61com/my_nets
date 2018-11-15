import sys
import Queue
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/..'.format(this_file_path))
from utils import Utils
from o2_load import *
from network import Network


class P1Net1(Network):

    def setup(self):
        pass

    def real_setup(self, nodes1, nodes2, post):
        self.feed('input_{}'.format(post))
        for a in range(len(nodes1) - 1):
            name = 'fc1_{}_{}'.format(post, a)
            self.fc(nodes1[a], name=name)
        self.fc(nodes1[-1], relu=False, name='reference_{}'.format(post))

        for a in range(len(nodes2) - 1):
            name = 'fc2_{}_{}'.format(post, a)
            self.fc(nodes2[a], name=name)
        self.fc(nodes2[-1], relu=False, name='output_{}'.format(post))

        print("number of layers = {} {} {}".format(len(self.layers), nodes1, nodes2))


class sNet3(Network):

    def setup(self):
        pass

    def real_setup(self, nodes):
        self.feed('data')
        for a in range(len(nodes) - 1):
            name = 'fc_{}'.format(a)
            self.fc(nodes[a], name=name)

        self.fc(nodes[-1], relu=False, name='output')

        print("number of layers = {} {}".format(len(self.layers), nodes))


class StackNet(Network):

    def parameters(self, stack, dim_input=4, dim_output=3, dim_ref=256):
        self.stack = stack
        self.dim_inter = [256]
        self.dim_ref = dim_ref
        self.dim_output = dim_output

        self.dim0 = dim_input + dim_ref
        self.out0 = self.dim_inter[0]
        self.weights0 = self.make_var('weights0', shape=[self.dim0, self.out0])
        self.biases0 = self.make_var('biases0', [self.out0])
        #
        # self.dim1 = self.out0
        # self.out1 = self.dim_inter[1]
        # self.weights1 = self.make_var('weights1', shape=[self.dim1, self.out1])
        # self.biases1 = self.make_var('biases1', [self.out1])

        self.dim2 = self.out0
        self.out2 = dim_ref
        self.weights2 = self.make_var('weights2', shape=[self.dim2, self.out2])
        self.biases2 = self.make_var('biases2', [self.out2])

        self.dim3 = self.out2
        self.out3 = self.dim_output
        self.weights3 = self.make_var('weights3', shape=[self.dim3, self.out3])
        self.biases3 = self.make_var('biases3', [self.out3])

    def setup(self):
        pass

    def real_setup(self, stack, verbose=True):
        self.parameters(stack)

        # base net
        (self.feed('input0').
         fc(2048, name='fc0').
         fc(self.dim_ref, name='fc1')
         .fc(self.dim_output, relu=False, name='output0')
         )

        ref_out_name = 'fc1'
        for a in range(self.stack):
            input_name = 'input{}'.format(a + 1)
            ic_name = 'ic{}_in'.format(a)
            ifc0_name = 'ifc0{}_in'.format(a)
            #ifc1_name = 'ifc1{}_in'.format(a)
            ifc2_name = 'ifc2{}_in'.format(a)
            output_name = 'output{}'.format(a + 1)

            (self.feed(input_name, ref_out_name)
             .concat(1, name=ic_name)
             .fc_w(name=ifc0_name,
                   weights=self.weights0,
                   biases=self.biases0)
             #.fc_w(name=ifc1_name,
             #      weights=self.weights1,
             #      biases=self.biases1)
             .fc_w(name=ifc2_name,
                   weights=self.weights2,
                   biases=self.biases2)
             .fc_w(name=output_name, relu=False,
                   weights=self.weights3,
                   biases=self.biases3)
             )
            # (self.feed(input_name, ref_out_name)
            #  .concat(1, name=ic_name)
            #  .fc(128, name=ifc0_name)
            #  .fc(64, name=ifc1_name)
            #  .fc(3, relu=False, name=output_name)
            #  )
            ref_out_name = ifc2_name

        if verbose:
            print("number of layers = {}".format(len(self.layers)))
            for l in sorted(self.layers.keys()):
                print l, self.layers[l].get_shape()


def _reshuffle(data):
    np.random.shuffle(data[0])


def _reshuffle_b(bucket):
    for data in bucket:
        np.random.shuffle(data[0])


class DataSet:
    def __init__(self, dataset, batch_size=500, npar=50, cache=True, nadd=0):
        self.bucket = 0
        self.dataset = dataset
        self.index = -1
        self.batch_size = batch_size
        self.nPar = npar
        self.nAdd = nadd
        self.data = None
        self.verbose = True
        self.cache = cache
        self.memories = {}

        self.load_next_data()
        self.sz = self.data[0][0][0].shape
        self.id = None

    def get_next(self, rd=True, num_output=3):
        self.load_next_data()

        if self.index == 0:
            return None
        return self.prepare(rd, num_output=num_output)

    def load_next_data(self):
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
            data = load_data(self.dataset[self.index], self.verbose)

        if self.cache:
            self.memories[self.index] = data

        self.data = []
        for a in range(0, len(data), self.batch_size):
            b = a + self.batch_size
            if b > len(data):
                b = len(data)
            self.data.append(data[a:b])

    def prepare_cnn(self, rd=False, num_output=3):
        pre_data = []
        self.reshuffle_data()
        for d in self.data:
            pre_data.append(self.create_bucket_cnn(d, num_output))
        if rd:
            np.random.shuffle(pre_data)
        return pre_data

    def prepare_stack(self, rd=True, num_output=3):
        pre_data = []
        self.reshuffle_data()
        self.id = 0
        for d in self.data:
            data = self.create_stack(d, num_output)
            if rd:
                np.random.shuffle(data)
            pre_data.append(data)

        return pre_data

    def create_stack(self, data, num_output):
        outputs = []

        for d in data:
            sz = d[0].shape
            if sz[0] < self.nPar + self.nAdd:
                d[0] = np.concatenate((d[0], d[0]))
                sz = d[0].shape
            truth = d[1][:num_output]
            d0 = d[0][:self.nPar].reshape(self.nPar * sz[1])
            d1 = d[0][self.nPar:]
            outputs.append([d0, d1, truth])

        return outputs

    def prepare_stage(self, rd=True, num_output=3, rdd=True):
        pre_data = []
        if rdd:
            self.reshuffle_data()
        self.id = 0
        for d in self.data:
            data = self.create_stage_data(d, num_output)
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

    def prepare(self, rd=True, num_output=3, multi=1, rdd=True):
        pre_data = []
        if rdd:
            self.reshuffle_data()
        self.id = 0
        for d in self.data:
            data = self.create_bucket(d, num_output, multi)
            if rd:
                np.random.shuffle(data)
            ins = []
            outs = []
            ids = []
            for a in data:
                ins.append(a[0])
                outs.append(a[1])
                ids.append(a[2])
            dd = (np.array(ins), np.array(outs), ids)
            pre_data.append(dd)

        return pre_data

    def create_stage_data(self, data, num_output):
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
                truth = d[1][:num_output]
                output = (i1, i2,  truth.reshape(num_output), self.id)
                outputs.append(output)

        self.id += 1

        return outputs


    def create_bucket(self, data, num_output, multi):
        if multi==-2:
            return self.create_stage_data(data, num_output)

        outputs = []
        # inputs = []
        # ids = []
        sz_in = data[0][0].shape

        for d in data:
            input = d[0]
            if multi > 0:
                num = multi  # *int(np.ceil(len(input)/float(self.nPar)))
            else:
                num = int(np.ceil(len(input) / float(self.nPar)))
            length = num * (self.nPar + self.nAdd)
            while len(input) < length:
                input = np.concatenate((input, input))
            input = input[:length]
            for a in range(0, len(input), self.nPar+self.nAdd):
                it = input[a:a + self.nPar]
                truth = d[1][:num_output]
                output = (it.reshape(self.nPar * sz_in[1]), truth.reshape(num_output), self.id)
                outputs.append(output)

            self.id += 1

        return outputs

    def create_bucket_cnn(self, data, num_output):
        outputs = []
        inputs = []
        sz_in = data[0][0].shape

        for d in data:
            input = d[0]
            while len(input) < self.nPar:
                input = np.concatenate((input, input))
            input = input[:self.nPar]
            truth = d[1][:num_output]
            inputs.append(input.reshape((self.nPar, sz_in[1], 1)))
            outputs.append(truth.reshape(num_output))

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
                inputs.append(input_p.reshape(self.nPar * sz_in[1]))
                outputs.append(output_p.reshape(num_output))

            pre_data.append((inputs, outputs))

        self.bucket = len(pre_data)
        return pre_data

    def gen_data(self, nPar, num_output):
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
    # ss = np.std(r, 1)
    # d0 = t[0]-mm[0]
    # d1 = t[1]-mm[1]
    dd = np.linalg.norm(t - mm)
    dd = dd * dd
    # dd = dd*dd

    return dd, mm


class sNet1(Network):

    def setup(self):
        (self.feed('data').
         fc(2048, name='fc1').
         fc(128, name='fc2').
         fc(1, relu=False, name='output'))

        print("number of layers = {}".format(len(self.layers)))


class sNet3_2(Network):

    def setup(self):
        pass

    def real_setup(self, nodes):
        self.feed('data0')
        for a in range(len(nodes) - 1):
            name = 'fc0_{}'.format(a)
            self.fc(nodes[a], name=name)
        self.fc(nodes[-1], relu=False, name='output0')

        print("number of layers = {} {}".format(len(self.layers), nodes))


class sNet3_stage(Network):

    def setup(self):
        pass

    def real_setup(self, nodes):
        self.feed('data1')
        for a in range(len(nodes) - 1):
            name = 'fc1_{}'.format(a)
            self.fc(nodes[a], name=name)
        self.fc(nodes[-1], relu=False, name='output1')

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


def run_data_stack(data, inputs, sess, xy, stack):
    results = None
    truth = None

    for b in data:
        length = b[0].shape[1] - 4 * stack
        feed = {inputs['input0']: b[0][:, :length]}
        for a in range(stack):
            feed[inputs['input{}'.format(a + 1)]] = b[0][:, length + 4 * a:length + 4 * (a + 1)]
        result = []
        for a in range(stack):
            r = sess.run(xy[a], feed_dict=feed)
            result.append(r)
        if results is None:
            results = np.array(result)
            truth = b[1]
        else:
            results = np.concatenate((results, np.array(result)), axis=1)
            truth = np.concatenate((truth, b[1]))

    return Utils.calculate_stack_loss(results, truth)


def run_data_stack_avg2(data, inputs, sess, xy, stack):
    rst_dic = {}
    truth_dic = {}
    for b in data:
        length = b[0].shape[1] - 4 * stack
        feed = {inputs['input0']: b[0][:, :length]}
        for a in range(stack):
            feed[inputs['input{}'.format(a + 1)]] = b[0][:, length + 4 * a:length + 4 * (a + 1)]
        result = []
        for a in range(stack+1):
            r = sess.run(xy[a], feed_dict=feed)
            result.append(r)
        result = np.array(result)
        for a in range(len(b[2])):
            if not b[2][a] in rst_dic:
                rst_dic[b[2][a]] = []
            rst_dic[b[2][a]].append(result[:, a, :])
            truth_dic[b[2][a]] = b[1][a]

    results = []
    truth = []

    filename = '/home/weihao/tmp/test.csv'
    if sys.platform == 'darwin':
        filename = '/Users/weihao/tmp/test.csv'
    fp = open(filename, 'w')
    for id in rst_dic:
        dst = np.array(rst_dic[id])
        result = np.median(dst, axis=0)
        results.append(result)
        truth.append(truth_dic[id])
        t = truth_dic[id]
        if random.random() < 0.2:
            r = np.linalg.norm(t - result)
            mm = result[stack - 1]
            fp.write('{},{},{},{},{},{},{}\n'.
                     format(t[0], mm[0], t[1], mm[1], t[2], mm[2], r))
    fp.close()

    results = np.array(results)
    truth = np.array(truth)
    L = []
    M = []
    for a in range(3):
        diff = results[:, 0:a+1,:].sum(axis=1) - truth
        r = np.linalg.norm(diff, axis=1)
        L.append((r*r).mean())
        M.append(np.median(r))
    return L, M


def run_data_stack_avg(data, inputs, sess, xy, stack):
    rst_dic = {}
    truth_dic = {}
    for b in data:
        length = b[0].shape[1] - 4 * stack
        feed = {inputs['input0']: b[0][:, :length]}
        for a in range(stack):
            feed[inputs['input{}'.format(a + 1)]] = b[0][:, length + 4 * a:length + 4 * (a + 1)]
        result = []
        for a in range(stack+1):
            r = sess.run(xy[a], feed_dict=feed)
            result.append(r)
        result = np.array(result)
        for a in range(len(b[2])):
            if not b[2][a] in rst_dic:
                rst_dic[b[2][a]] = []
            rst_dic[b[2][a]].append(result[:, a, :])
            truth_dic[b[2][a]] = b[1][a]

    results = []
    truth = []

    filename = '/home/weihao/tmp/test.csv'
    if sys.platform == 'darwin':
        filename = '/Users/weihao/tmp/test.csv'
    fp = open(filename, 'w')
    for id in rst_dic:
        dst = np.array(rst_dic[id])
        result = np.median(dst, axis=0)
        results.append(result)
        truth.append(truth_dic[id])
        t = truth_dic[id]
        if random.random() < 0.2:
            r = np.linalg.norm(t - result)
            mm = result[stack - 1]
            fp.write('{},{},{},{},{},{},{}\n'.
                     format(t[0], mm[0], t[1], mm[1], t[2], mm[2], r))
    fp.close()

    return Utils.calculate_stack_loss_avg(np.array(results), np.array(truth))


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


def run_stage_data(data,
                   input0,
                   input1,
                   sess,
                   xy,
                   xy_ref,
                   xy_stage,
                   st_ref
                   ):
    results = None
    truth = None
    stage_result = None

    for b in data:
        feed = {input0: b[0]}
        result = sess.run(xy, feed_dict=feed)
        if results is None:
            results = result
            truth = b[2]
        else:
            results = np.concatenate((results, result))
            truth = np.concatenate((truth, b[2]))

        fc_input = sess.run(xy_ref, feed_dict=feed)
        b1 = b[1]
        n1 = b1.shape[1]
        for a in range(n1):
            input_array = np.concatenate((fc_input, b1[:, a, :]), axis=1)
            feed = {input1: input_array}

            if a == n1-1:
                r = sess.run(xy_stage, feed_dict=feed)
                if stage_result is None:
                    stage_result = r
                else:
                    stage_result = np.concatenate((stage_result, r))
                break
            else:
                fc_input = sess.run(st_ref, feed_dict=feed)

        #print results.shape, truth.shape, stage_result.shape

    a, b = Utils.calculate_loss(results, truth)
    avg = Utils.calculate_loss(stage_result, truth-results)

    return a, b, avg
