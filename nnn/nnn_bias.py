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
        self.active_function = relu
        self.num_layers = len(layers)
        self.learning_rate = lr

        number = self.input_dim + 1
        for a in range(self.num_layers):
            layer = layers[a]
            self.weights.append(np.random.randn(number, layer)/np.sqrt(number))

            number = layer + 1

        self.output_weights = np.random.randn(number, self.output_dim)/np.sqrt(number)

    def train(self, inputs, outputs, num_steps=1):

        pre_loss = 0
        for t in range(num_steps):

            Zs = [add_1(inputs)]
            for a in range(self.num_layers):
                A = Zs[a].dot(self.weights[a])
                Z = add_1(self.active_function(A))
                Zs.append(Z)

            predicts = Z.dot(self.output_weights)# + self.output_bias
            grad = outputs - predicts

            loss = np.square(grad).sum()

            #if pre_loss is not None and t % 50 == 0:
            # print '{0} {1} {2}'.format(t, loss, pre_loss - loss)

            pre_loss = loss

            grad *= 2
            D_output_weights = Zs[-1].T.dot(grad)

            grad = grad.dot(self.output_weights.T)
            grad = grad * self.active_function(Zs[-1], True)

            D_weight_m1 = Zs[-2].T.dot(grad[:, :-1])


            self.output_weights += D_output_weights*self.learning_rate

            self.weights[-1] += D_weight_m1*self.learning_rate

            #self.bias[-2] += D_bias_m2*self.learning_rate
            #self.weights[-2] += D_weight_m2*self.learning_rate

    def run(self,inputs):
        Z = add_1(inputs)
        for a in range(self.num_layers):
            A = Z.dot(self.weights[a])
            Z = add_1(self.active_function(A))
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
    #from sklearn import datasets
    #X, y = datasets.make_moons(16, noise=0.10)

    np.random.seed(1)

    N, D_in, D_out = 64, 1000, 10
    X = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    D_in = X.shape[1]
    D_out = y.shape[1]
    nnn = NNNS(D_in, D_out, [100])

    nnn.train(X, y, 100)
