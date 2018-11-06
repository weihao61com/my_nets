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

    def setup(self, lr, init):
        self.base_nn.setup(lr, init)
        self.stack_nn.setup(lr, init)
        self.final_nn.setup(lr, init)

    def reset(self):
        s1 = self.base_nn.reset()
        s2 = self.stack_nn.reset()
        s3 = self.final_nn.reset()
        return s1 + ' ' + s2 + ' ' + s3

    def run(self, inputs):
        outputs = []
        for a in inputs:
            output, _ = self._run(a)
            outputs.append(output[0])
        return outputs

    def _run(self, inputs):
        if inputs.shape[1] < self.feture * self.attribute:
            raise Exception("data is too short {} vs {}".
                            format(len(inputs), self.feture * self.attribute))
        Zs = []
        ref, Z = self.base_nn.run(inputs[:, -self.feture * self.attribute:])
        Zs.append(Z)
        for a in range(0, inputs.shape[1], self.attribute):
            ref = np.concatenate((ref, inputs[:, a:a + self.attribute]), axis=1)
            ref, Z = self.stack_nn.run(ref)
            Zs.append(Z)
            break

        output, Z = self.final_nn.run(ref)
        Zs.append(Z)
        return output, Zs

    def backward(self, grad, Zs):

        grad = self.final_nn.backward(grad, Zs[-1], True)

        for a in range(len(Zs)-2):
            grad = self.stack_nn.backward(grad, Zs[-2-a], True)
            grad = grad[:, :-self.attribute]

        self.base_nn.backward(grad, Zs[0])

    def train(self, inputs, outputs):
        loss = 0
        for a in range(len(inputs)):
            loss += self._train(inputs[a], outputs[a])
        return loss

    def _train(self, inputs, outputs):
        predicts, Zs = self._run(inputs)
        grad = outputs - predicts
        self.backward(grad, Zs)
        return (grad*grad).sum()

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

    reference = js["nodes_reference"]
    base_nodes = map(int, js["nodes_base"].split(','))
    stack_nodes = map(int, js["nodes_stack"].split(','))
    final_nodes = map(int, js["nodes_final"].split(','))

    num_att = 4
    num_output = 3

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '.p'

    tr = DataSet(tr_data, batch_size)
    te_set = DataSet(te_data, batch_size)

    sz_in = te_set.sz
    iterations = 10000
    loop = 1000
    print "input shape", sz_in, "LR", lr, 'feature', feature_len

    if renetFile is not None:
        with open(renetFile, 'r') as fp:
            stack = pickle.load(fp)
        stack.setup(lr, False)

    else:
        base_nn = NNN(feature_len * num_att, reference, base_nodes, True)
        stack_nn = NNN(reference + num_att, reference, stack_nodes, True)
        final_nn = NNN(reference, num_output, final_nodes)
        stack = Stack(base_nn, stack_nn, final_nn, feature_len, num_att)
        stack.setup(lr, True)

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

        lt0 = datetime.datetime.now()
        loss = 0
        length = 0
        for t in range(loop):
            str1 = stack.reset()

            tr_pre_data = tr.prepare_stack()
            while tr_pre_data:
                for b in tr_pre_data:
                    loss += stack.train(b[0], b[1])
                    length += len(b[0])
                tr_pre_data = tr.get_next()
            if t%10 == 0:
                print 'its', t+a*loop, loss/length, str1, datetime.datetime.now()-lt0
                loss = 0
                length = 0
                lt0 = datetime.datetime.now()

        with open(netFile, 'w') as fp:
            pickle.dump(stack, fp)
