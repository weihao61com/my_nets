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
from fc_dataset import DataSet, Config


def add_1(inputs):
    return np.concatenate((inputs, np.array([[1] * len(inputs)]).T), axis=1)


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def relu(x, derivative=False):
    return 1. * (x > 0) if derivative else np.maximum(x, 0)


class NNN:

    def __init__(self, input_dim, output_dim, layers, final_layer_act=False, af='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if af=='relu':
            self.active_function = relu
        else:
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

            #if a==0:
            #    print d0[1][2],self.gradient_momentum[a][1][2],\
            #        np.sqrt(self.g2_momentum[a][1][2])

    def reset(self):

        p = self.gradient_momentum
        a = self.g2_momentum
        w = self.weights
        output = '{0:.6f} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f}'. \
            format(p[0][1][2], np.sqrt(a[0][1][2]), w[0][1][2],
                   p[1][1][2], np.sqrt(a[1][1][2]), w[1][1][2]
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
        rst_dic = {}
        truth_dic = {}
        for b in data:
            inputs = b[0]
            truth = b[1]
            id = b[2]
            result, _ = self.run(inputs)

            for a in range(len(id)):
                if id[a] not in rst_dic:
                    rst_dic[id[a]] = []
                    truth_dic[id[a]] = truth[a]
                rst_dic[id[a]].append(result[a])


        results = []
        truth = []

        for id in rst_dic:
            rst = np.median(np.array(rst_dic[id]), axis=0)
            results.append(rst)
            truth.append(truth_dic[id])

        return Utils.calculate_loss(np.array(results), np.array(truth))


def run_testing(te, nnn):
    te_pre_data = te.prepare(multi=-1)
    loss, median = nnn.run_data(te_pre_data)
    print loss, median


if __name__ == '__main__':

    config_file = "config.json"

    if len(sys.argv)>1:
        config_file = sys.argv[1]

    cfg = Config(config_file)
    batch_size = cfg.batch_size

    tr = DataSet(cfg.tr_data, cfg.memory_size, cfg.feature_len)
    te = DataSet(cfg.te_data, cfg.memory_size, cfg.feature_len)
    tr.set_net_type(cfg.net_type)
    te.set_net_type(cfg.net_type)
    tr.set_t_scale(cfg.t_scale)
    te.set_t_scale(cfg.t_scale)
    tr.set_num_output(cfg.num_output)
    te.set_num_output(cfg.num_output)

    att = te.sz[1]
    iterations = 100000

    print "input shape", att, "LR", cfg.lr, 'feature', cfg.feature_len

    testing = None
    if len(sys.argv) > 2:
        if sys.argv[2].startswith('te'):
            testing = te
        else:
            testing = tr

    att = te.sz[1]
    iterations = 10000
    loop = cfg.loop
    print "input shape", att, "LR", cfg.lr, 'feature', cfg.feature_len

    D_in = cfg.feature_len * att
    D_out = cfg.num_output
    netFile = cfg.netFile[:-3] + '.p'

    if cfg.renetFile is not None:
        renetFile = cfg.renetFile[:-3] + '.p'
        with open(renetFile, 'r') as fp:
            nnn = pickle.load(fp)
        nnn.setup(cfg.lr, init=False)
    else:
        nnn = NNN(D_in, D_out, cfg.nodes[0], af=cfg.af)
        nnn.setup(cfg.lr)

    if testing:
        run_testing(te, nnn)
        exit(0)

    t00 = datetime.datetime.now()
    str1 = ''
    for a in range(iterations):
        tr_pre_data = tr.prepare()
        total_loss, tr_median = nnn.run_data(tr_pre_data)

        te_pre_data = te.prepare()
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
            tr_pre_data = tr.prepare()
            while tr_pre_data:
                for b in tr_pre_data:
                    for c in range(0, len(b[2]), batch_size):

                        loss += nnn.train(b[0][c:c + batch_size], b[1][c:c + batch_size])
                        length += len(b[0][c:c + batch_size])

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