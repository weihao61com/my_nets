
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import numpy as np

HOME = '/home/weihao/Projects/'
if sys.platform=='darwin':
    HOME = '/Users/weihao/Projects/'

sys.path.append('{}/my_nets'.format(HOME))
sys.path.append('{}/my_nets/fc'.format(HOME))

from utils import Utils
from fc_dataset import Config, DataSet


class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

class MySigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        sigmoid_eval = 1.0/(1.0 + torch.exp(-input))
        ctx.save_for_backward(sigmoid_eval)
        return sigmoid_eval

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # grad_temp = grad_output.clone()
        # grad_input = torch.exp(-input)/((1.0 + torch.exp(-input)).pow(2))
        return grad_output*input*(1-input)


class Net(nn.Module):

    def __init__(self, af):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(80, 100)
        self.fc2 = nn.Linear(100, 100)
        # self.fc3 = nn.Linear(100, 100)
        # self.fc4 = nn.Linear(100, 100)
        # self.fc5 = nn.Linear(100, 100)
        # self.fc6 = nn.Linear(100, 100)
        # self.fc7 = nn.Linear(100, 100)
        # self.fc8 = nn.Linear(100, 100)
        self.fc9 = nn.Linear(100, 50)
        self.fc10 = nn.Linear(50, 3)
        self.af = af

    def forward(self, x):
        #sigmoid = nn.Sigmoid()
        # F = MySigmoid.apply
        # F = MyReLU.apply
        # F = nn.Sigmoid()
        F = self.af
        x = F(self.fc1(x))
        x = F(self.fc2(x))
        # x = F(self.fc3(x))
        # x = F(self.fc4(x))
        # x = F(self.fc5(x))
        # x = F(self.fc6(x))
        # x = F(self.fc7(x))
        # x = F(self.fc8(x))
        x = F(self.fc9(x))
        x = self.fc10(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config', required=False)
    parser.add_argument('-t', '--test', help='test', required=False)
    args = parser.parse_args()

    config_file = "config.json" if args.config is None else args.config
    mode = 'train' if args.test is None else args.test

    iterations = 10000
    js = Utils.load_json_file(config_file)

    #F = MyReLU.apply
    #lr = 0.38e-4

    #F = MySigmoid.apply
    F = nn.Sigmoid()
    lr = 1.0e-4

    net = Net(F)
    print(net)
    #
    # params = list(net.parameters())
    # print(len(params))
    # print(params[0].size())  # conv1's .weight
    #
    input = torch.randn(2000, 80)
    # out = net(input)
    # #print(out)
    #
    # net.zero_grad()
    target  = torch.randn(2000, 3)
    # out.backward(target)
    #
    # output = net(input)
    # #target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    import torch.optim as optim

    # create your optimizer
    # optimizer = optim.SGD(net.parameters(), lr=0.0015)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # in your training loop:
    ll=0
    NN=0
    aa=0.0
    for a in range(iterations):
        optimizer.zero_grad()  # zero the gradient buffers
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        ll += loss.data.numpy()
        NN += 1
        aa += a
        if NN==100:
            print aa/NN, ll/NN
            if ll/NN <1e-4:
                break
            ll=0
            NN=0
            aa=0.0