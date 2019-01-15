import torch
import sys
import numpy as np
import os
import datetime

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/..'.format(this_file_path))

from dataset import DataSet, Config
from utils import Utils



class FC(nn.Module):
    def __init__(self, n_input, nodes, n_output,  learning_rate=.02):
        super(FC, self).__init__()

        self.activeLayer = F.relu
        self.n_input = n_input
        self.layers = []

        ins = n_input
        nt = 0
        for n in nodes:
            layer = nn.Linear(ins, n)
            self.__setattr__('L_{}'.format(nt), layer)
            self.layers.append(layer)
            ins = n
            nt += 1

        layer = nn.Linear(ins, n_output)
        self.__setattr__('O', layer)
        self.output = layer

        self.criterion = nn.MSELoss()

        for p in self.parameters():
            print p.data.size()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, data):

        for l in self.layers:
            data = self.activeLayer(l(data))


        return self.output(data)

    def train(self, outs, ins):
        error = 0.0

        for a in range(ins.size()[0]):

            self.zero_grad()
            output = self(ins[a])
            loss = self.criterion(output, outs[a])
            error += loss.detach().numpy()*output.size()[0]

            loss.backward()

            self.optimizer.step()

        return error/ins.size()[0]

    def run(self, ins):

        outputs = []
        for a in range(ins.size()[0]):
            b = ins[a]
            o0 = []
            for i in range(b.size()[0]):
                output = self(b[i])
            outputs.append(output)

        return np.array(outputs)


def ToTensor(array):
    return torch.tensor(array)


def run_test(mrnn, tr, cfg, hidden):
    rst_dic = {}
    truth_dic = {}

    tr_pre_data = tr.prepare(multi=-1)
    while tr_pre_data:
        for b in tr_pre_data:
            length = len(b[0])
            x = ToTensor(b[0].reshape(length, cfg.feature_len, cfg.att).astype(np.float32))
            outputs = mrnn.run(x, hidden)
            for a in range(len(b[2])):
                if not b[2][a] in rst_dic:
                    rst_dic[b[2][a]] = []
                rst_dic[b[2][a]].append(outputs[a, :, :])
                truth_dic[b[2][a]] = b[1][a]

        results = []
        truth = []

        #filename = '/home/weihao/tmp/{}.csv'.format(fname)
        #if sys.platform == 'darwin':
        #    filename = '/Users/weihao/tmp/{}.csv'.format(fname)
        #fp = open(filename, 'w')
        for id in rst_dic:
            dst = np.array(rst_dic[id])
            result = np.median(dst, axis=0)
            results.append(result)
            truth.append(truth_dic[id])
            # if random.random() < 0.2:
            #     r = np.linalg.norm(t - result)
            #     mm = result[-1]
            #     if len(mm) == 3:
            #         fp.write('{},{},{},{},{},{},{}\n'.
            #                  format(t[0], mm[0], t[1], mm[1], t[2], mm[2], r))
            #     else:
            #         fp.write('{},{},{}\n'.
            #                  format(t[0], mm[0], r))
        #fp.close()
        tr_pre_data = tr.get_next()

    return Utils.calculate_stack_loss_avg(np.array(results), np.array(truth))



def main(args):

    config_file = args.config
    test = args.test

    cfg = Config(config_file)

    tr = None
    if test is None:
        tr = DataSet(cfg.tr_data, cfg)
        te = DataSet(cfg.te_data, cfg, sub_sample=1)
        tr0 = DataSet([cfg.tr_data[0]], cfg, sub_sample=1)
        cfg.att = te.sz[1]
    else:
        if test == 'te':
            te = DataSet([cfg.te_data[0]], cfg)
        else:
            te = DataSet([cfg.tr_data[0]], cfg)
        cfg.att = te.sz[1]

    iterations = 10000
    loop = cfg.loop
    print "input attribute", cfg.att, "LR", cfg.lr, 'feature', cfg.feature_len

    n_att = cfg.att
    # n_length = cfg.feature_len
    n_hidden = cfg.nodes[1][-1]
    n_output = cfg.num_output
    hidden0 = ToTensor(np.ones(n_hidden).astype(np.float32))

    mrnn = FC(n_att*cfg.feature_len, cfg.nodes[0], n_output, cfg.lr)

    if test:
        mrnn.load_state_dict(torch.load(cfg.netTest[:-3]))
        run_test(mrnn, te, cfg, hidden0)
        tr_loss, tr_median = run_test(mrnn, te, cfg, hidden0)
        for a in range(len(tr_loss)):
            print a, tr_loss[a], tr_median[a]

        exit(0)

    if cfg.renetFile:
        mrnn.load_state_dict(torch.load(cfg.renetFile[:-3]))

    t00 = datetime.datetime.now()

    T = 0
    T_err=0
    for a in range(iterations):

        tr_pre_data = tr.prepare(multi=1)
        while tr_pre_data:
            for b in tr_pre_data:
                x = ToTensor(b[0].astype(np.float32))
                y = ToTensor(b[1].astype(np.float32))
                err = mrnn.train(y, x)
                if a%loop==0 and a>0:
                    t1 = datetime.datetime.now()
                    print a, (t1 - t00).total_seconds()/3600.0, T_err/T
                    T_err=0
                    T = 0
                    torch.save(mrnn.state_dict(), cfg.netFile[:-3])
                T_err += err
                T += 1

            tr_pre_data = tr.get_next()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config', default='fc_t.json')
    parser.add_argument('-t', '--test', help='test', default=None)
    args = parser.parse_args()

    main(args)