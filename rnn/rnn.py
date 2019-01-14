import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self, n_att, n_length, n_hidden, n_output,  learning_rate=.02):
        super(RNN, self).__init__()

        self.hidden_size = n_hidden
        self.length = n_length
        self.att = n_att

        h0 = 256

        self.i2h0 = nn.Linear(n_att + n_hidden, h0)
        self.i2h1 = nn.Linear(h0, n_hidden)

        self.i2o0 = nn.Linear(n_att + n_hidden, h0)
        self.i2o1 = nn.Linear(h0, n_output)
        # self.relu = F.relu
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden):

        # combined = torch.cat((input, hidden), 1)
        combined = torch.cat((input, hidden))

        hidden0 = self.i2h0(combined)
        hidden1 = self.i2h1(hidden0)

        output0 = self.i2o0(combined)
        output1 = self.i2o1(output0)

        return output1, hidden1

    def train(self, outs, ins, hidden0):


        error = 0.0
        self.zero_grad()

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
