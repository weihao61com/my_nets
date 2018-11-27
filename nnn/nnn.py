import numpy as np
import sys
import datetime
import pickle
import os
import shutil

HOME = '{}/Projects/'.format(os.getenv('HOME'))
sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils
from fc_dataset import DataSet


def add_1(inputs):
    return np.concatenate((inputs, np.array([[1] * len(inputs)]).T), axis=1)


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def relu(x, derivative=False):
    return 1. * (x > 0) if derivative else np.maximum(x, 0)


class NNN:

    def __init__(self, input_dim, output_dim, layers, final_layer_act=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.active_function = sigmoid
        layers.append(output_dim)
        self.num_layers = len(layers)
        self.final_act = final_layer_act

        self.weights = []
        self.D_weight = []
        self.gradient_momentum = []
        self.g2_momentum = []

        self.learning_rate = None
        self.beta1 = None
        self.eps_stable= None

        number = self.input_dim + 1
        for a in range(self.num_layers):
            layer = layers[a]
            w = np.random.randn(number, layer) / np.sqrt(number)
            self.weights.append(w)
            number = layer + 1

    def setup(self, lr, init=True):
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.eps_stable = 1e-8

        self.learning_rate = lr

        if init:
            self.gradient_momentum = []
            self.g2_momentum = []
            self.D_weight = []
            for a in range(self.num_layers):
                w = self.weights[a]
                self.gradient_momentum.append(np.zeros(w.shape))
                self.g2_momentum.append(np.zeros(w.shape))
                self.D_weight.append(np.zeros(w.shape))

    def backward(self, grad, Zs, with_grad=False):
        #n1 = np.linalg.norm(grad)
        if self.final_act:
            grad = grad * self.active_function(Zs[-1], True)

        self.D_weight[-1] = Zs[-2].T.dot(grad)
        weigh_T = self.weights[-1].T
        for a in range(self.num_layers - 1):
            grad = grad.dot(weigh_T)
            grad = grad * self.active_function(Zs[-2 - a], True)
            grad = grad[:, :-1]
            self.D_weight[-2 - a] = Zs[-3 - a].T.dot(grad)
            weigh_T = self.weights[-2 - a].T

        self.update_momentum()
        for a in range(self.num_layers):
            v2 = np.sqrt(self.g2_momentum[a]) + self.eps_stable
            div = self.gradient_momentum[a] / v2

            self.weights[a] += div * self.learning_rate

        if not with_grad:
            return None

        grad = grad.dot(weigh_T)[:, :-1]
        # grad = grad * self.active_function(Zs[-1 - a], True)
        # print n1, np.linalg.norm(grad)
        return grad

    def train(self, inputs, outputs):

        predicts, Zs = self.run(inputs)
        grad = outputs - predicts
        loss = np.square(grad).sum()
        self.backward(grad, Zs)

        return loss

    def update_momentum(self):
        for a in range(self.num_layers):

            d0 = self.D_weight[a]
            #sign = (d>0)*2-1
            #d0 = np.maximum(abs(d), 1e-6) * sign

            v1 = self.beta1 * self.gradient_momentum[a] \
                + (1 - self.beta1) * d0

            v2 = self.beta2 * self.g2_momentum[a] \
                 + (1 - self.beta2) * d0 * d0

            self.gradient_momentum[a] = v1
            self.g2_momentum[a] = v2

    def reset(self):

        p = self.gradient_momentum
        a = self.g2_momentum
        w = self.weights
        output = '{0:.6f} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f}'. \
            format(p[0][1][2]*100, np.sqrt(a[0][1][2])*100, w[0][1][2]*100,
                   p[1][1][2]*100, np.sqrt(a[1][1][2])*100, w[1][1][2]*100
               )

        return output

    def run(self, inputs):
        Z = add_1(inputs)
        Zs = [Z]
        for a in range(self.num_layers - 1):
            A = Z.dot(self.weights[a])
            Z = add_1(self.active_function(A))
            Zs.append(Z)

        A = Z.dot(self.weights[-1])
        if self.final_act:
            A = self.active_function(A)

        Zs.append(A)
        return A, Zs

    def run_data(self, data):
        results = None
        truth = None

        for b in data:
            inputs = b[0]
            result, _ = self.run(inputs)
            if results is None:
                results = result
                truth = b[1]
            else:
                results = np.concatenate((results, result))
                truth = np.concatenate((truth, b[1]))

        return Utils.calculate_loss(results, truth)


if __name__ == '__main__':

    config_file = "config.json"

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
    batch_size = int(js['batch_size'])
    feature_len = int(js['feature'])
    lr = float(js['lr'])

    num_output = int(js["num_output"])
    nodes = map(int, js["nodes"].split(','))

    renetFile = None
    if 'retrain' in js:
        renetFile = HOME + 'NNs/' + js['retrain'] + '.p'

    tr = DataSet(tr_data, batch_size, feature_len)
    te = DataSet(te_data, batch_size, feature_len)
    tr.set_num_output(num_output)
    te.set_num_output(num_output)

    sz_in = te.sz
    iterations = 10000
    loop = 100
    print "input shape", sz_in, "LR", lr, 'feature', feature_len

    D_in = feature_len * sz_in[1]
    D_out = num_output

    if renetFile is not None:
        with open(renetFile, 'r') as fp:
            nnn = pickle.load(fp)
    else:
        nnn = NNN(D_in, D_out, nodes)

    nnn.setup(lr)

    t00 = datetime.datetime.now()
    str1 = ''
    for a in range(iterations):
        tr_pre_data = tr.prepare(multi=1)
        total_loss, tr_median = nnn.run_data(tr_pre_data)

        te_pre_data = te.prepare(multi=1)
        te_loss, te_median = nnn.run_data(te_pre_data)

        t1 = (datetime.datetime.now() - t00).seconds / 3600.0
        str = "iteration: {0} {1:.3f} {2:.4f} {3:.4f} {4:.4f} {5:.4f} ".format(
            a * loop / 1000.0, t1, total_loss, te_loss,
            tr_median, te_median)
        print str + str1
        #t00 = t1

        loss = 0
        length = 0
        lt0 = datetime.datetime.now()

        for t in range(loop):
            str1 = nnn.reset()
            tr_pre_data = tr.prepare(multi=1)
            while tr_pre_data:
                for b in tr_pre_data:
                    loss += nnn.train(b[0], b[1])
                    length += len(b[0])

                tr_pre_data = tr.get_next()
        str1 = '{0:.4f}  {1}'.format(loss/length, str1)
            #if t%10 == 0:
            #    print 'its', t+a*loop, loss/length, str1, datetime.datetime.now()-lt0
            #    loss = 0
            #    length = 0
            #    lt0 = datetime.datetime.now()

        tmp_file = netFile+'.tmp'
        with open(tmp_file, 'w') as fp:
            pickle.dump(nnn, fp)
        shutil.copy(tmp_file, netFile)
        os.remove(tmp_file)