
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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(80, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 3)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #x = F.max_pool2d(MyReLU.apply(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        #x = F.max_pool2d(MyReLU.apply(self.conv2(x)), 2)
        #x = x.view(-1, self.num_flat_features(x))
        x = MyReLU.apply(self.fc1(x))
        x = MyReLU.apply(self.fc2(x))
        x = MyReLU.apply(self.fc3(x))
        x = self.fc4(x)
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

    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight

    input = torch.randn(2000, 80)
    out = net(input)
    #print(out)

    net.zero_grad()
    target  = torch.randn(2000, 3)
    out.backward(target)

    output = net(input)
    #target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)

    #print(loss.grad_fn)  # MSELoss
    #print(loss.grad_fn.next_functions[0][0])  # Linear
    #print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


    net.zero_grad()     # zeroes the gradient buffers of all parameters

    #print('conv1.bias.grad before backward')
    #print(net.conv1.bias.grad)

    loss.backward()

    #print('conv1.bias.grad after backward')
    #print(net.conv1.bias.grad)

    import torch.optim as optim

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update
    print loss
