import numpy as np
import sys

sys.path.append('..')
sys.path.append('../fc')
from utils import Utils
from fc_dataset import *
import datetime
import pickle

from nnn_bias import NNNB

def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def relu(x, derivative=False):
    return 1. * (x > 0) if derivative else np.maximum(x, 0)


class NNN:

    def __init__(self, input_dim, output_dim, layers, lr = 1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = []
        self.bias = []
        self.active_function = sigmoid
        self.num_layers = len(layers)
        self.learning_rate = lr

        number = self.input_dim
        for a in range(self.num_layers):
            layer = layers[a]
            self.weights.append(np.random.randn(number, layer)/np.sqrt(number))
            #self.bias.append(np.random.randn(layer))
            number = layer

        self.output_weights = np.random.randn(number, self.output_dim)/np.sqrt(number)
        #self.output_bias = np.random.randn(self.output_dim)

    def train(self, inputs, outputs):

        Zs = [inputs]
        for a in range(self.num_layers):
            A = Zs[a].dot(self.weights[a]) #+ self.bias[a]
            Z = self.active_function(A)
            Zs.append(Z)

        predicts = Z.dot(self.output_weights) #+ self.output_bias
        grad = outputs - predicts

        loss = np.square(grad).sum()

        grad *= 2
        #D_output_bias = sum(grad)
        D_output_weights = Zs[-1].T.dot(grad)

        grad = grad.dot(self.output_weights.T)
        grad = grad * self.active_function(Zs[-1], True)

        #D_bias_m1 = sum(grad)
        D_weight_m1 = Zs[-2].T.dot(grad)


        #self.output_bias += D_output_bias*self.learning_rate
        self.output_weights += D_output_weights*self.learning_rate

        #self.bias[-1] += D_bias_m1*self.learning_rate
        self.weights[-1] += D_weight_m1*self.learning_rate

        return loss

    def run(self,inputs):
        Z = inputs
        for a in range(self.num_layers):
            A = Z.dot(self.weights[a])
            Z = self.active_function(A)
        predicts = Z.dot(self.output_weights)
        return predicts

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

        return Utils.calculate_loss(results, truth)


if __name__ == '__main__':

    HOME = '/home/weihao/Projects/'
    if sys.platform == 'darwin':
        HOME = '/Users/weihao/Projects/'

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
    te_set = DataSet(te_data, batch_size, feature_len)

    sz_in = te_set.sz
    iterations = 10000
    loop = 100
    print "input shape", sz_in, "LR", lr, 'feature', feature_len

    D_in = feature_len* sz_in[1]
    D_out = num_output

    if renetFile is not None:
        with open(renetFile, 'r') as fp:
            nnn = pickle.load(fp)
    else:
        nnn = NNNB(D_in, D_out, nodes, lr=lr)

    nnn.setup()

    t00 = datetime.datetime.now()
    str1 = ''
    for a in range(iterations):
        tr_pre_data = tr.prepare(multi=1)
        total_loss, tr_median = nnn.run_data(tr_pre_data)

        te_pre_data = te_set.prepare(multi=1)
        te_loss, te_median = nnn.run_data(te_pre_data)

        t1 = datetime.datetime.now()
        str = "iteration: {} {} {} {} {} time {} ".format(
            a * loop, total_loss, te_loss,
            tr_median, te_median, t1 - t00)
        print str + str1
        t00 = t1

        for t in range(loop):
            str1 = nnn.reset()
            loss = 0
            tr_pre_data = tr.prepare(multi=1)
            while tr_pre_data:
                for b in tr_pre_data:
                    loss += nnn.train(b[0], b[1])
                tr_pre_data = tr.get_next()
            #print t, loss

        with open(netFile, 'w') as fp:
            pickle.dump(nnn, fp)