"""
Masked Autoregressive Flow for Density Estimation
arXiv:1705.07057v4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import math
import argparse
import pprint
import copy

from data import fetch_dataloaders


parser = argparse.ArgumentParser()
# action
parser.add_argument('--train', action='store_true', help='Train a flow.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a flow.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--data_dir', default='./data/', help='Location of datasets.')
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
# data
parser.add_argument('--dataset', default='toy', help='Which dataset to use.')
parser.add_argument('--flip_toy_var_order', action='store_true', help='Whether to flip the toy dataset variable order to (x2, x1).')
parser.add_argument('--seed', type=int, default=1, help='Random seed to use.')
# model
parser.add_argument('--model', default='maf', help='Which model to use: made, maf.')
# made parameters
parser.add_argument('--n_blocks', type=int, default=5, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=100, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--activation_fn', type=str, default='relu', help='What activation function to use in the MADEs.')
parser.add_argument('--input_order', type=str, default='sequential', help='What input order to use (sequential | random).')
parser.add_argument('--conditional', default=False, action='store_true', help='Whether to use a conditional model.')
parser.add_argument('--no_batch_norm', action='store_true')
# training params
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--log_interval', type=int, default=1000, help='How often to show loss statistics and save samples.')


# --------------------
# Model layers and helpers
# --------------------

def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]

    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [input_degrees - 1]

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    """ MADE building block layer """
    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer('mask', mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) + (self.cond_label_size != None) * ', cond_features={}'.format(self.cond_label_size)


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """
    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = - (1 - self.mask) * s  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du

        return x, log_abs_det_jacobian


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0) # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
#        print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
#            (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians

# --------------------
# Models
# --------------------

class MADE(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', input_degrees=None):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2,1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = - loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        D = u.shape[1]
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
            x[:,i] = u[:,i] * torch.exp(loga[:,i]) + m[:,i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)


class MADEMOG(nn.Module):
    """ Mixture of Gaussians MADE """
    def __init__(self, n_components, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', input_degrees=None):
        """
        Args:
            n_components -- scalar; number of gauassian components in the mixture
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        self.n_components = n_components

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, n_components * 3 * input_size, masks[-1].repeat(n_components * 3,1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # shapes
        N, L = x.shape
        C = self.n_components
        # MAF eq 2 -- parameters of Gaussians - mean, logsigma, log unnormalized cluster probabilities
        m, loga, logr = self.net(self.net_input(x, y)).view(N, C, 3 * L).chunk(chunks=3, dim=-1)  # out 3 x (N, C, L)
        # MAF eq 4
        x = x.repeat(1, C).view(N, C, L)  # out (N, C, L)
        u = (x - m) * torch.exp(-loga)  # out (N, C, L)
        # MAF eq 5
        log_abs_det_jacobian = - loga  # out (N, C, L)
        # normalize cluster responsibilities
        self.logr = logr - logr.logsumexp(1, keepdim=True)  # out (N, C, L)
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # shapes
        N, C, L = u.shape
        # init output
        x = torch.zeros(N, L).to(u.device)
        # MAF eq 3
        # run through reverse model along each L
        for i in self.input_degrees:
            m, loga, logr = self.net(self.net_input(x, y)).view(N, C, 3 * L).chunk(chunks=3, dim=-1)  # out 3 x (N, C, L)
            # normalize cluster responsibilities and sample cluster assignments from a categorical dist
            logr = logr - logr.logsumexp(1, keepdim=True)  # out (N, C, L)
            z = D.Categorical(logits=logr[:,:,i]).sample().unsqueeze(-1)  # out (N, 1)
            u_z = torch.gather(u[:,:,i], 1, z).squeeze()  # out (N, 1)
            m_z = torch.gather(m[:,:,i], 1, z).squeeze()  # out (N, 1)
            loga_z = torch.gather(loga[:,:,i], 1, z).squeeze()
            x[:,i] = u_z * torch.exp(loga_z) + m_z
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)  # u = (N,C,L); log_abs_det_jacobian = (N,C,L)
        # marginalize cluster probs
        log_probs = torch.logsumexp(self.logr + self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)  # sum over C; out (N, L)
        return log_probs.sum(1)  # sum over L; out (N,)


class MAF(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', batch_norm=True):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [MADE(input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)


class MAFMOG(nn.Module):
    """ MAF on mixture of gaussian MADE """
    def __init__(self, n_blocks, n_components, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu',
                 input_order='sequential', batch_norm=True):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        self.maf = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, batch_norm)
        # get reversed input order from the last layer (note in maf model, input_degrees are already flipped in for-loop model constructor
        input_degrees = self.maf.input_degrees#.flip(0)
        self.mademog = MADEMOG(n_components, input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, input_degrees)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        u, maf_log_abs_dets = self.maf(x, y)
        u, made_log_abs_dets = self.mademog(u, y)
        sum_log_abs_det_jacobians = maf_log_abs_dets.unsqueeze(1) + made_log_abs_dets
        return u, sum_log_abs_det_jacobians

    def inverse(self, u, y=None):
        x, made_log_abs_dets = self.mademog.inverse(u, y)
        x, maf_log_abs_dets = self.maf.inverse(x, y)
        sum_log_abs_det_jacobians = maf_log_abs_dets.unsqueeze(1) + made_log_abs_dets
        return x, sum_log_abs_det_jacobians

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)  # u = (N,C,L); log_abs_det_jacobian = (N,C,L)
        # marginalize cluster probs
        log_probs = torch.logsumexp(self.mademog.logr + self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)  # out (N, L)
        return log_probs.sum(1)  # out (N,)


class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)


# --------------------
# Train and evaluate
# --------------------

def train(model, dataloader, optimizer, epoch, args):

    for i, data in enumerate(dataloader):
        model.train()

        # check if labeled dataset
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
            y = y.to(args.device)
        x = x.view(x.shape[0], -1).to(args.device)

        loss = - model.log_prob(x, y if args.cond_label_size else None).mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch, args.start_epoch + args.n_epochs, i, len(dataloader), loss.item()))

@torch.no_grad()
def evaluate(model, dataloader, epoch, args):
    model.eval()

    # conditional model
    if args.cond_label_size is not None:
        logprior = torch.tensor(1 / args.cond_label_size).log().to(args.device)
        loglike = [[] for _ in range(args.cond_label_size)]

        for i in range(args.cond_label_size):
            # make one-hot labels
            labels = torch.zeros(args.batch_size, args.cond_label_size).to(args.device)
            labels[:,i] = 1

            for x, y in dataloader:
                x = x.view(x.shape[0], -1).to(args.device)
                loglike[i].append(model.log_prob(x, labels))

            loglike[i] = torch.cat(loglike[i], dim=0)   # cat along data dim under this label
        loglike = torch.stack(loglike, dim=1)           # cat all data along label dim

        # log p(x) = log ∑_y p(x,y) = log ∑_y p(x|y)p(y)
        # assume uniform prior      = log p(y) ∑_y p(x|y) = log p(y) + log ∑_y p(x|y)
        logprobs = logprior + loglike.logsumexp(dim=1)
        # TODO -- measure accuracy as argmax of the loglike

    # unconditional model
    else:
        logprobs = []
        for data in dataloader:
            x = data[0].view(data[0].shape[0], -1).to(args.device)
            logprobs.append(model.log_prob(x))
        logprobs = torch.cat(logprobs, dim=0).to(args.device)

    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
    output = 'Evaluate ' + (epoch != None)*'(epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std)
    print(output)
    print(output, file=open(args.results_file, 'a'))
    return logprob_mean, logprob_std

@torch.no_grad()
def generate(model, dataset_lam, args, step=None, n_row=10):
    model.eval()

    # conditional model
    if args.cond_label_size:
        samples = []
        labels = torch.eye(args.cond_label_size).to(args.device)

        for i in range(args.cond_label_size):
            # sample model base distribution and run through inverse model to sample data space
            u = model.base_dist.sample((n_row, args.n_components)).squeeze()
            labels_i = labels[i].expand(n_row, -1)
            sample, _ = model.inverse(u, labels_i)
            log_probs = model.log_prob(sample, labels_i).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
            samples.append(sample[log_probs])

        samples = torch.cat(samples, dim=0)

    # unconditional model
    else:
        u = model.base_dist.sample((n_row**2, args.n_components)).squeeze()
        samples, _ = model.inverse(u)
        log_probs = model.log_prob(samples).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
        samples = samples[log_probs]

    # convert and save images
    samples = samples.view(samples.shape[0], *args.input_dims)
    samples = (torch.sigmoid(samples) - dataset_lam) / (1 - 2 * dataset_lam)
    filename = 'generated_samples' + (step != None)*'_epoch_{}'.format(step) + '.png'
    save_image(samples, os.path.join(args.output_dir, filename), nrow=n_row, normalize=True)

def train_and_evaluate(model, train_loader, test_loader, optimizer, args):
    best_eval_logprob = float('-inf')

    for i in range(args.start_epoch, args.start_epoch + args.n_epochs):
        train(model, train_loader, optimizer, i, args)
        eval_logprob, _ = evaluate(model, test_loader, i, args)

        # save training checkpoint
        torch.save({'epoch': i,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()},
                    os.path.join(args.output_dir, 'model_checkpoint.pt'))
        # save model only
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_state.pt'))

        # save best state
        if eval_logprob > best_eval_logprob:
            best_eval_logprob = eval_logprob
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                        os.path.join(args.output_dir, 'best_model_checkpoint.pt'))

        # plot sample
        if args.dataset == 'TOY':
            plot_sample_and_density(model, train_loader.dataset.base_dist, args, step=i)
        if args.dataset == 'MNIST':
            generate(model, train_loader.dataset.lam, args, step=i)

# --------------------
# Plot
# --------------------

def plot_density(dist, ax, ranges, flip_var_order=False):
    (xmin, xmax), (ymin, ymax) = ranges
    # sample uniform grid
    n = 200
    xx1 = torch.linspace(xmin, xmax, n)
    xx2 = torch.linspace(ymin, ymax, n)
    xx, yy = torch.meshgrid(xx1, xx2)
    xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze()

    if flip_var_order:
        xy = xy.flip(1)

    # run uniform grid through model and plot
    density = dist.log_prob(xy).exp()
    ax.contour(xx, yy, density.view(n,n).data.numpy())

    # format
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([xmin, xmax])
    ax.set_yticks([ymin, ymax])

def plot_dist_sample(data, ax, ranges):
    ax.scatter(data[:,0].data.numpy(), data[:,1].data.numpy(), s=10, alpha=0.4)
    # format
    (xmin, xmax), (ymin, ymax) = ranges
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([xmin, xmax])
    ax.set_yticks([ymin, ymax])

def plot_sample_and_density(model, target_dist, args, ranges_density=[[-5,20],[-10,10]], ranges_sample=[[-4,4],[-4,4]], step=None):
    model.eval()
    fig, axs = plt.subplots(1, 2, figsize=(6,3))

    # sample target distribution and pass through model
    data = target_dist.sample((2000,))
    u, _ = model(data)

    # plot density and sample
    plot_density(model, axs[0], ranges_density, args.flip_var_order)
    plot_dist_sample(u, axs[1], ranges_sample)

    # format and save
    matplotlib.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'sample' + (step != None)*'_epoch_{}'.format(step) + '.png'))
    plt.close()



# --------------------
# Run
# --------------------

if __name__ == '__main__':

    args = parser.parse_args()

    # setup file ops
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # setup device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # load data
    if args.conditional: assert args.dataset in ['MNIST', 'CIFAR10'], 'Conditional inputs only available for labeled datasets MNIST and CIFAR10.'
    train_dataloader, test_dataloader = fetch_dataloaders(args.dataset, args.batch_size, args.device, args.flip_toy_var_order)
    args.input_size = train_dataloader.dataset.input_size
    args.input_dims = train_dataloader.dataset.input_dims
    args.cond_label_size = train_dataloader.dataset.label_size if args.conditional else None

    # model
    if args.model == 'made':
        model = MADE(args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                     args.activation_fn, args.input_order)
    elif args.model == 'mademog':
        assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
        model = MADEMOG(args.n_components, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                     args.activation_fn, args.input_order)
    elif args.model == 'maf':
        model = MAF(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                    args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm)
    elif args.model == 'mafmog':
        assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
        model = MAFMOG(args.n_blocks, args.n_components, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                    args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm)
    elif args.model =='realnvp':
        model = RealNVP(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                        batch_norm=not args.no_batch_norm)
    else:
        raise ValueError('Unrecognized model.')

    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    if args.restore_file:
        # load model and optimizer states
        state = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        args.start_epoch = state['epoch'] + 1
        # set up paths
        args.output_dir = os.path.dirname(args.restore_file)
    args.results_file = os.path.join(args.output_dir, args.results_file)

    print('Loaded settings and model:')
    print(pprint.pformat(args.__dict__))
    print(model)
    print(pprint.pformat(args.__dict__), file=open(args.results_file, 'a'))
    print(model, file=open(args.results_file, 'a'))

    if args.train:
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, args)

    if args.evaluate:
        evaluate(model, test_dataloader, None, args)

    if args.generate:
        if args.dataset == 'TOY':
            base_dist = train_dataloader.dataset.base_dist
            plot_sample_and_density(model, base_dist, args, ranges_density=[[-15,4],[-3,3]], ranges_sample=[[-1.5,1.5],[-3,3]])
        elif args.dataset == 'MNIST':
            generate(model, train_dataloader.dataset.lam, args)


