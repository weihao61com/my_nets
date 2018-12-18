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
        return '' #s1 + ' ' + s2 + ' ' + s3

    def run(self, inputs):
        outputs = self._run(inputs)
        return outputs[0]

    def _run(self, inputs):
        if inputs.shape[1] < self.feture * self.attribute:
            raise Exception("data is too short {} vs {}".
                            format(len(inputs), self.feture * self.attribute))
        Zs = []
        Zo = []
        Zr = []
        ref, Z = self.base_nn.run(inputs)
        # ref, Z = self.base_nn.run(inputs[:, -self.feture * self.attribute:])
        output, r = self.final_nn.run(ref)
        Zo.append(output)
        Zr.append(r)
        Zs.append(Z)
        for a in range(0, inputs.shape[1], self.attribute):
            ref = np.concatenate((ref, inputs[:, a:a + self.attribute]), axis=1)
            ref, Z = self.stack_nn.run(ref)
            output, r = self.final_nn.run(ref)
            Zo.append(output)
            Zr.append(r)
            Zs.append(Z)

        return Zo, Zr, Zs

    def _train(self, inputs, outputs):
        Zo, Zr, Zs = self._run(inputs)
        grad = outputs - Zo[-1]
        self.backward(grad, Zr[-1], Zs, Zs[0])
        return (grad*grad).sum()

    def backward(self, grad, Z0, Zs, Zfinal):

        grad = self.final_nn.backward(grad, Z0, True)

        for a in range(len(Zs)-2):
            grad = self.stack_nn.backward(grad, Zs[-1-a], True)
            grad = grad[:, :-self.attribute]

        # print np.linalg.norm(grad)
        self.base_nn.backward(grad, Zfinal)

    def train(self, inputs, outputs):
        #loss = 0
        #for a in range(len(inputs)):
        #    loss += self._train(inputs[a], outputs[a])
        return self._train(inputs, outputs)

    def run_data(self, data):

        rst_dic = {}
        tru_dic = {}

        for b in data:
            result = self.run(b[0])[-1]
            for a in range(len(b[2])):
                id = b[2][a]
                if not id in rst_dic:
                    rst_dic[id] = []
                rst_dic[id].append(result[a, :])
                tru_dic[id] = b[1][a]

        results = []
        truth = []

        for id in rst_dic:
            dst = np.array(rst_dic[id])
            result = np.median(dst, axis=0)
            results.append(result)
            truth.append(tru_dic[id])

        return Utils.calculate_loss(np.array(results), np.array(truth))

    def run_data_stack(self, data):

        rst_dic = {}
        tru_dic = {}

        for b in data:
            result = np.array(self.run(b[0]))
            for a in range(len(b[2])):
                id = b[2][a]
                if not id in rst_dic:
                    rst_dic[id] = []
                rst_dic[id].append(result[:, a, :])
                tru_dic[id] = b[1][a]

        results = []
        truth = []

        for id in rst_dic:
            dst = np.array(rst_dic[id])
            result = np.median(dst, axis=0)
            results.append(result)
            truth.append(tru_dic[id])

        results = np.array(results)
        truth = np.array(truth)
        M = []
        L = []
        for a in range(results.shape[1]):
            l, m = Utils.calculate_loss(results[:, a, :], truth)
            M.append(m)
            L.append(l)
        return L, M

def run_test(stack, te):
    L, M = stack.run_data_stack(te.prepare(multi=-1))
    for a in range(len(L)):
        print a, L[a], M[a]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config', required=False)
    parser.add_argument('-t', '--test', help='test', required=False)
    args = parser.parse_args()

    config_file = "cstack.json" if args.config is None else args.config
    mode = 'train' if args.test is None else args.test

    iterations = 10000
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
    final_nodes = [] #map(int, js["nodes_final"].split(','))

    num_att = 4
    num_output = 3

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '.p'

    if mode == 'train':
        tr = DataSet(tr_data, js['block'], feature_len)
        # tr.set_t_scale(t_scale)
        tr.set_num_output(num_output)
        te = DataSet(te_data, js['block'], feature_len)
        # te.set_t_scale(t_scale)
        te.set_num_output(num_output)
    else:
        if mode == 'te':
            te = DataSet(te_data, js['block'], feature_len)
        else:
            te = DataSet(tr_data, js['block'], feature_len)
        te.set_num_output(num_output)

    sz_in = te.sz
    loop = js['loop']
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

    if mode=='train':

        t00 = datetime.datetime.now()
        str1 = ''
        for a in range(iterations):
            total_loss, tr_median = stack.run_data(tr.prepare())
            te_loss, te_median = stack.run_data(te.prepare())
            t1 = datetime.datetime.now()
            str = "it: {0:.3f} {1:.3f} {2:.5f} {3:.5f} {4:.5f} {5:.5f}".format(a * loop / 1000.0,
                                               (t1 - t00).total_seconds() / 3600.0,
                                               total_loss, te_loss,
                                               tr_median, te_median,
                                               )
            print str + str1
            loss = 0
            length = 0
            for t in range(loop):
                #str1 = stack.reset()
                #tr_pre_data = tr.prepare_stack()
                tr_pre_data = tr.prepare(multi=1)
                while tr_pre_data:
                    for b in tr_pre_data:
                        for c in range(0, len(b[2]), batch_size):
                            loss += stack.train(b[0][c:c+batch_size], b[1][c:c+batch_size])
                        length += len(b[0])
                    tr_pre_data = tr.get_next()

            str1 = ' {0:.5f}'.format(loss/length)
            with open(netFile, 'w') as fp:
                pickle.dump(stack, fp)
    else:
        run_test(stack, te)
