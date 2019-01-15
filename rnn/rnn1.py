import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


class RNN(nn.Module):
    def __init__(self, n_att, nodes, n_output,  learning_rate=.02):
        super(RNN, self).__init__()

        self.activeLayer = F.relu
        # self.nodes = nodes
        self.att = n_att

        self.feature = []
        self.recurrent = []
        self.output = []

        ref = nodes[1][-1]

        ins = n_att
        nt = 0
        for n in nodes[0]:
            layer = nn.Linear(ins, n)
            self.__setattr__('L0_{}'.format(nt), layer)
            self.feature.append(layer)
            ins = n
            nt += 1

        ins += ref
        nt = 0
        for n in nodes[1]:
            layer = nn.Linear(ins, n)
            self.__setattr__('L1_{}'.format(nt), layer)
            self.recurrent.append(layer)
            ins = n
            nt += 1

        for n in nodes[2]:
            layer = nn.Linear(ins, n)
            self.__setattr__('L2_{}'.format(nt), layer)
            self.output.append(layer)
            ins = n
            nt += 1

        layer = nn.Linear(ins, n_output)
        self.__setattr__('O', layer)
        self.output.append(layer)

        #print n_att, nodes[0]
        #print nodes[1]
        #print nodes[2], n_output
        #
        # self.i2h0 = nn.Linear(n_att + n_hidden, h0)
        # self.i2h1 = nn.Linear(h0, n_hidden)
        #
        # self.i2o0 = nn.Linear(n_att + n_hidden, h0)
        # self.i2o1 = nn.Linear(h0, n_output)
        # self.relu = F.relu
        self.criterion = nn.MSELoss()

        for p in self.parameters():
            print p.data.size()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden):

        for l in self.feature:
            input = l(input)

        combined = torch.cat((input, hidden))
        for l in self.recurrent:
            combined = l(combined)

        output = combined
        for l in self.output:
            output = l(output)

        return output, combined

    def train(self, outs, ins, hidden0):
        error = 0.0

        for a in range(ins.size()[0]):

            self.zero_grad()
            hidden = hidden0
            b = ins[a]
            touts = []
            for i in range(b.size()[0]):
                output, hidden = self(b[i], hidden)
                touts.append(output)

            loss = self.criterion(output, outs[a])
            error += loss.detach().numpy()*output.size()[0]
            #
            loss.backward()
            # #loss.backward(retain_graph=True)
            #
            # # Add parameters' gradients to their values, multiplied by learning rate
            # for p in self.parameters():
            #     p.data.add_(-learning_rate, p.grad.data)
            self.optimizer.step()

        return error/ins.size()[0]

    def run(self, ins, hidden0):

        outputs = []
        for a in range(ins.size()[0]):
            hidden = hidden0
            b = ins[a]
            o0 = []
            for i in range(b.size()[0]):
                output, hidden = self(b[i], hidden)
                o0.append(output.detach().numpy())
            outputs.append(o0[-2:])

        return np.array(outputs)
