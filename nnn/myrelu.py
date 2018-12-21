import torch
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

    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    cfg = Config(config_file)

    tr = DataSet(cfg.tr_data, cfg.memory_size, cfg.feature_len)
    te = DataSet(cfg.te_data, cfg.memory_size, cfg.feature_len)
    tr.set_net_type(cfg.net_type)
    te.set_net_type(cfg.net_type)
    tr.set_t_scale(cfg.t_scale)
    te.set_t_scale(cfg.t_scale)
    tr.set_num_output(cfg.num_output)
    te.set_num_output(cfg.num_output)
    att = te.sz[1]

    D_in = cfg.feature_len * att
    D_out = cfg.num_output

    tr_pre_data = tr.prepare(multi=1)
    for b in tr_pre_data:
        d = torch.from_numpy(b[0]).type(torch.FloatTensor)
        t = torch.from_numpy(b[1]).type(torch.FloatTensor)

    te_pre_data = te.prepare(multi=1)
    for b in te_pre_data:
        de = torch.from_numpy(b[0]).type(torch.FloatTensor)
        te = torch.from_numpy(b[1]).type(torch.FloatTensor)

    H = 200

    # Create random Tensors to hold input and outputs.
    #d = torch.randn(N, D_in, device=device, dtype=dtype)
    #t = torch.randn(N, D_out, device=device, dtype=dtype)

    # Create random Tensors for weights.
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
    learning_rate = 1e-5

    for a in range(iterations):
        # To apply our Function, we use Function.apply method. We alias this as 'relu'.
        relu = MyReLU.apply

        # Forward pass: compute predicted y using operations; we compute
        # ReLU using our custom autograd operation.
        y_pred = relu(d.mm(w1)).mm(w2)
        loss = (y_pred - t).pow(2)

        if a%500==0:
            l0 = loss.data.numpy()
            ye_pred = relu(de.mm(w1)).mm(w2)
            losse = (ye_pred - te).pow(2)
            l1 = losse.data.numpy()
            l0 = np.mean(l0, axis=1)
            l1 = np.mean(l1, axis=1)

            print '{0:6d} {1:9.1f} {2:9.2f}  {3:9.2f} {4:9.2f}'.format(
                a, np.mean(l0), np.mean(l1), np.sqrt(np.median(l0)), np.sqrt(np.median(l1)))


        loss = loss.mean()

        # Use autograd to compute the backward pass.
        loss.backward()

        # Update weights using gradient descent
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()