import sys
import os
import pickle
import datetime
import numpy as np
from nnn import NNN

HOME = '{}/Projects/'.format(os.getenv('HOME'))
sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils
from fc_dataset import DataSet


class Stack:
    def __init__(self, n1, n2, n3, feature, attribute):
        self.base_nn = n1
        self.stack_nn = n2
        self.final_nn = n3
        self.feture = feature
        self.attribute = attribute

    def setup(self, lr):
        self.base_nn.setup(lr)
        self.stack_nn.setup(lr)
        self.final_nn.setup(lr)

    def reset(self):
        self.base_nn.reset()
        self.stack_nn.reset()
        self.final_nn.reset()

    def run(self, inputs):
        outputs = []
        for a in inputs:
            output, _ = self._run(a)
            outputs.append(output[0])
        return outputs

    def _run(self, input):
        if input.shape[1]<self.feture*self.attribute:
            raise Exception("data is too short {} vs {}".
                            format(len(input), self.feture*self.attribute))
        ref, _ = self.base_nn.run(input[:, -self.feture*self.attribute:])
        for a in range(0, len(input), self.attribute):
            ref = np.concatenate((ref, input[:, a:a+self.attribute]), axis=1)
            ref, _ = self.stack_nn.run(ref)
        return self.final_nn.run(ref)

    def run_data(self, data):
        results = None
        truth = None

        for b in data:
            inputs = b[0]
            result = self.run(inputs)
            if results is None:
                results = result
                truth = b[1]
            else:
                results = np.concatenate((results, result))
                truth = np.concatenate((truth, b[1]))

        return Utils.calculate_loss(np.array(results), np.array(truth))


if __name__ == '__main__':

    config_file = "cstack.json"

    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    js = Utils.load_json_file(config_file)

    tr_data = []
    te_data = []
    for key in js:
        if key.startswith('tr'):
            tr_data.append(HOME + js[key])
        if key.startswith('te'):
            te_data.append(HOME + js['te'])

    netFile = HOME + 'NNs/' + js['net'] + '.p'
    batch_size = js['batch_size']
    lr = js['lr']
    feature_len = js['feature']

    reference = js["reference"]
    base_nodes =  map(int, js["base_nodes"].split(','))
    stack_nodes = map(int, js["stack_nodes"].split(','))
    final_nodes = map(int, js["final_nodes"].split(','))

    num_att = 4
    num_output = 3

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '.p'

    tr = DataSet(tr_data, batch_size)
    te_set = DataSet(te_data, batch_size)

    sz_in = te_set.sz
    iterations = 10000
    loop = 4
    print "input shape", sz_in, "LR", lr, 'feature', feature_len


    if renetFile is not None:
        with open(renetFile, 'r') as fp:
            stack = pickle.load(fp)
    else:
        base_nn = NNN(feature_len*num_att, reference, base_nodes)
        stack_nn = NNN(reference+num_att, reference, stack_nodes)
        final_nn = NNN(reference, num_output, final_nodes)
        stack = Stack(base_nn, stack_nn, final_nn, feature_len, num_att)

    stack.setup(lr)

    t00 = datetime.datetime.now()
    str1 = ''
    for a in range(iterations):
        tr_pre_data = tr.prepare_stack()
        total_loss, tr_median = stack.run_data(tr_pre_data)

        te_pre_data = te_set.prepare_stack()
        te_loss, te_median = stack.run_data(te_pre_data)

        t1 = datetime.datetime.now()
        str = "iteration: {0} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5} ".format(
            a * loop, total_loss, te_loss,
            tr_median, te_median, t1 - t00)
        print str + str1
        t00 = t1

        for t in range(loop):
            str1 = stack.reset()
            loss = 0
            tr_pre_data = tr.prepare_stack()
            while tr_pre_data:
                for b in tr_pre_data:
                    loss += nnn.train(b[0], b[1])
                tr_pre_data = tr.get_next()
            # print t, loss

        with open(netFile, 'w') as fp:
            pickle.dump(nnn, fp)
