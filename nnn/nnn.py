import numpy as np
from random import random


def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))


class NNN:

    def __index__(self, inputs, outputs, layers):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = []
        self.bias = []
        self.active_function = sigmoid
        self.layer = len(layers)

        num_input = self.inputs
        for a in range(len(layers)):
            layer = layers[a]
            self.weights.append(2 * random((num_input, layer)) - 1)
            # self.bias.append(2 * random(layer) - 1)
            num_input = layer

        self.output_weights = 2 * random((num_input, self.outputs)) - 1
        # self.output_bias = 2 * random(self.outputs) - 1

    def train(self, inputs, outputs, num_steps):

        for t in range(num_steps):

            layers = [inputs]
            for a in range(self.layer):
                layers.append(self.active_function(layers[a].dot(self.weights[a])))

            grad = layers[-1] - outputs

            loss = np.square(grad).sum()
            if t % 50 == 0:
                print t, loss

            for a in range(self.layer):
                diff = layers[-1-a].T.dot(grad)
                w1 -= learning_rate * grad_w1

            grad_w2 = h_relu.T.dot(grad_y_pred)
            grad_h_relu = grad_y_pred.dot(w2.T)
            grad_h = grad_h_relu.copy()
            if relu:
                grad_h[h < 0] = 0
            grad_w1 = x.T.dot(grad_h)

            # Update weights
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2

            # Find delta, i.e. Product of Error and derivative of next layer
            delta = outputError * self.sigmoid_derivative(output)

            # Multiply with transpose of last layer
            # to invert the multiplication we did to get layer
            out_weights_adjustment = dot(layer3.T, delta)

            # Apply the out_weights_adjustment
            self.out_weights += out_weights_adjustment

            # Procedure stays same, but the error now is the product of current weight and
            # Delta in next layer
            delta = dot(delta, self.out_weights.T) * self.sigmoid_derivative(layer3)
            weight_3_adjustment = dot(layer2.T, delta)
            self.weights_3 += weight_3_adjustment

            delta = dot(delta, self.weights_3.T) * self.sigmoid_derivative(layer2)
            weight_2_adjustment = dot(layer1.T, delta)
            self.weights_2 += weight_2_adjustment

            delta = dot(delta, self.weights_2.T) * self.sigmoid_derivative(layer1)
            weight_1_adjustment = dot(self.gateInput.T, delta)
            self.weights_1 += weight_1_adjustment


if __name__ == '__main__':

    N, D_in, D_out = 512, 100, 10

    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    nnn = NNN(D_in, D_out, [4,8])
