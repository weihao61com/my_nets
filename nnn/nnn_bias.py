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
        self.beta1 = None
        self.D_weight = []
        self.gradient_momentum = []

        number = self.input_dim + 1
        for a in range(self.num_layers):
            layer = layers[a]
            w = np.random.randn(number, layer)/np.sqrt(number)
            self.weights.append(w)
            self.gradient_momentum.append(np.zeros(w.shape))
            self.D_weight.append(np.zeros(w.shape))
            number = layer + 1

    def setup(self):
        self.beta1 = 0.999

    def train(self, inputs, outputs):

        Z = None
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
            #self.D_weight[-1-a] += D_weight_m[a]
            self.D_weight[-1-a] = D_weight_m[a]

        self.update_momentum()

        for a in range(self.num_layers):
            self.weights[a] += self.gradient_momentum[a]*self.learning_rate

        return loss

    def update_momentum(self):
        for a in range(self.num_layers):
            self.gradient_momentum[a] = self.beta1 * self.gradient_momentum[a]\
                                        + (1 - self.beta1) * self.D_weight[a]

    def reset(self):

        p = self.gradient_momentum
        w = self.weights
        #s = self.D_weight
        output = '{0:.6f} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f} '.\
            format(p[0][1][10], p[1][10][21], p[2][15][0],
                   w[0][1][10], w[1][10][21], w[2][15][0])
        # print output

        self.D_weight = []
        for a in range(self.num_layers):
            self.D_weight.append(np.zeros(self.weights[a].shape))
        return output

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
