import numpy as np
import sys

sys.path.append('..')
from utils import Utils

def add_1(inputs):
    return np.concatenate ((inputs, np.array([[1]*len(inputs)]).T), axis=1)


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def relu(x, derivative=False):
    return 1. * (x > 0) if derivative else np.maximum(x, 0)


class NNNB:

    def __init__(self, input_dim, output_dim, layers, lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = []
        self.active_function = sigmoid
        layers.append(output_dim)
        self.num_layers = len(layers)
        self.learning_rate = lr

        self.D_weight = None
        self.D_pre_weight = None
        self.learning_scale = []

        number = self.input_dim + 1
        for a in range(self.num_layers):
            layer = layers[a]
            w = np.random.randn(number, layer)/np.sqrt(number)
            self.weights.append(w)
            number = layer + 1

    def train(self, inputs, outputs):

        Zs = [add_1(inputs)]
        for a in range(self.num_layers - 1):
            A = Zs[a].dot(self.weights[a])
            Z = add_1(self.active_function(A))
            Zs.append(Z)

        predicts = Z.dot(self.weights[-1])
        grad = outputs - predicts

        loss = np.square(grad).sum()

        D_weight_m = [Zs[-1].T.dot(grad)]

        weigh_T = self.weights[-1].T

        for a in range(self.num_layers - 1):
            grad = grad.dot(weigh_T)
            grad = grad * self.active_function(Zs[-1-a], True)
            grad = grad[:, :-1]
            D_weight_m.append(Zs[-2-a].T.dot(grad))
            weigh_T = self.weights[-2-a].T

        for a in range(self.num_layers):
            self.D_weight[-1-a] += D_weight_m[a]
            for b in range(len(D_weight_m[a])):
                self.weights[-1-a][b] += D_weight_m[a][b]*\
                                         self.learning_rate*\
                                         self.learning_scale[a][b]

        return loss

    def reset(self, first=False):
        if first:
            self.D_pre_weight = None
            self.D_weight = None
            for a in range(self.num_layers):
                w = self.weights[self.num_layers - 1 - a]
                self.learning_scale.append(np.ones(w.shape))

        if self.D_pre_weight is not None:
            p = self.D_pre_weight
            w = self.weights
            s = self.learning_scale
            print 'A{0:12.6f} {1:12.6f} {2:5.2f} ' \
                  '{3:12.6f} {4:12.6f} {5:5.2f} ' \
                  '{6:12.6f} {7:12.6f} {8:5.2f}'.\
                format(p[0][1][10],w[0][1][10],s[2][1][10],
                       p[1][10][21],w[1][10][21],s[1][10][21],
                       p[2][15][0], w[2][15][0], s[0][15][0])
            for a in range(self.num_layers):
                p = self.D_pre_weight[a]
                w = self.D_weight[a]
                s = self.learning_scale[-1-a]
                for b in range(len(p)):
                    for c in range(len(p[b])):
                        if np.sign(p[b][c]) == np.sign(w[b][c]):
                            if np.abs(p[b][c]) < np.abs(w[b][c]):
                                s[b][c] *= 1.1
                        else:
                            s[b][c] *= 0.5
                            if s[b][c]<0.01:
                                s[b][c] = 0.01

        self.D_pre_weight = self.D_weight
        self.D_weight = []
        for a in range(self.num_layers):
            self.D_weight.append(np.zeros(self.weights[a].shape))

    def run(self,inputs):
        Z = add_1(inputs)
        for a in range(self.num_layers - 1):
            A = Z.dot(self.weights[a])
            Z = add_1(self.active_function(A))
        predicts = Z.dot(self.weights[-1])
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
    #from sklearn import datasets
    #X, y = datasets.make_moons(16, noise=0.10)

    np.random.seed(1)

    N, D_in, D_out = 64, 1000, 10
    X = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    D_in = X.shape[1]
    D_out = y.shape[1]
    nnn = NNNB(D_in, D_out, [100])

    nnn.train(X, y, 100)
