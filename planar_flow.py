"""
Variational Inference with Normalizing Flows
arXiv:1505.05770v6
"""

import torch
import torch.nn as nn
import torch.distributions as D

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse


parser = argparse.ArgumentParser()
# action
parser.add_argument('--train', action='store_true', help='Train a flow.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a flow.')
parser.add_argument('--plot', action='store_true', help='Plot a flow and target density.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--output_dir', default='.', help='Path to output folder.')
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
# target potential
parser.add_argument('--target_potential', choices=['u_z0', 'u_z5', 'u_z1', 'u_z2', 'u_z3', 'u_z4'], help='Which potential function to approximate.')
# flow params
parser.add_argument('--base_sigma', type=float, default=4, help='Std of the base isotropic 0-mean Gaussian distribution.')
parser.add_argument('--learn_base', default=False, action='store_true', help='Whether to learn a mu-sigma affine transform of the base distribution.')
parser.add_argument('--flow_length', type=int, default=2, help='Length of the flow.')
# training params
parser.add_argument('--init_sigma', type=float, default=1, help='Initialization std for the trainable flow parameters.')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--start_step', type=int, default=0, help='Starting step (if resuming training will be overwrite from filename).')
parser.add_argument('--n_steps', type=int, default=1000000, help='Optimization steps.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay.')
parser.add_argument('--beta', type=float, default=1, help='Multiplier for the target potential loss.')
parser.add_argument('--seed', type=int, default=2, help='Random seed.')


# --------------------
# Flow
# --------------------

class PlanarTransform(nn.Module):
    def __init__(self, init_sigma=0.01):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))

    def forward(self, x, normalize_u=True):
        # allow for a single forward pass over all the transforms in the flows with a Sequential container
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0

        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        u_hat = self.u
        if normalize_u:
            wtu = (self.w @ self.u.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())

        # compute transform
        f_z = z + u_hat * torch.tanh(z @ self.w.t() + self.b)
        # compute log_abs_det_jacobian
        psi = (1 - torch.tanh(z @ self.w.t() + self.b)**2) @ self.w
        det = 1 + psi @ u_hat.t()
        log_abs_det_jacobian = torch.log(torch.abs(det) + 1e-6).squeeze()
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian

        return f_z, sum_log_abs_det_jacobians

class AffineTransform(nn.Module):
    def __init__(self, learnable=False):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(2)).requires_grad_(learnable)
        self.logsigma = nn.Parameter(torch.zeros(2)).requires_grad_(learnable)

    def forward(self, x):
        z = self.mu + self.logsigma.exp() * x
        sum_log_abs_det_jacobians = self.logsigma.sum()
        return z, sum_log_abs_det_jacobians


# --------------------
# Test energy functions -- NF paper table 1
# --------------------

w1 = lambda z: torch.sin(2 * math.pi * z[:,0] / 4)
w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:,0] - 1)/0.6)**2)
w3 = lambda z: 3 * torch.sigmoid((z[:,0] - 1) / 0.3)

u_z1 = lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4)**2 - \
                 torch.log(torch.exp(-0.5*((z[:,0] - 2) / 0.6)**2) + torch.exp(-0.5*((z[:,0] + 2) / 0.6)**2) + 1e-10)
u_z2 = lambda z: 0.5 * ((z[:,1] - w1(z)) / 0.4)**2
u_z3 = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w2(z))/0.35)**2) + 1e-10)
u_z4 = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.4)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w3(z))/0.35)**2) + 1e-10)


# --------------------
# Training
# --------------------

def optimize_flow(base_dist, flow, target_energy_potential, optimizer, args):

    # anneal rate for free energy
    temp = lambda i: min(1, 0.01 + i/10000)

    for i in range(args.start_step, args.n_steps):

        # sample base dist
        z = base_dist.sample((args.batch_size, )).to(args.device)

        # pass through flow:
        # 1. compute expected log_prob of data under base dist -- nothing tied to parameters here so irrelevant to grads
        base_log_prob = base_dist.log_prob(z)
        # 2. compute sum of log_abs_det_jacobian through the flow
        zk, sum_log_abs_det_jacobians = flow(z)
        # 3. compute expected log_prob of z_k the target_energy potential
        p_log_prob = - temp(i) * target_energy_potential(zk)  # p = exp(-potential) ==> p_log_prob = - potential

        loss = base_log_prob - sum_log_abs_det_jacobians - args.beta * p_log_prob
        loss = loss.mean(0)

        # compute loss and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10000 == 0:
            # display loss
            log_qk = base_dist.log_prob(z) - sum_log_abs_det_jacobians
            print('{}: step {:5d} / {}; loss {:.3f}; base_log_prob {:.3f}, sum log dets {:.3f}, p_log_prob {:.3f}, max base = {:.3f}; max qk = {:.3f} \
                zk_mean {}, zk_sigma {}; base_mu {}, base_log_sigma {}'.format(
                args.target_potential, i, args.n_steps, loss.item(), base_log_prob.mean(0).item(), sum_log_abs_det_jacobians.mean(0).item(),
                p_log_prob.mean(0).item(), base_log_prob.exp().max().item(), log_qk.exp().max().item(),
                zk.mean(0).cpu().data.numpy(), zk.var(0).sqrt().cpu().data.numpy(),
                base_dist.loc.cpu().data.numpy() if not args.learn_base else flow[0].mu.cpu().data.numpy(),
                base_dist.covariance_matrix.cpu().diag().data.numpy() if not args.learn_base else flow[0].logsigma.cpu().data.numpy()))

            # save model
            torch.save({'step': i,
                        'flow_state': flow.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                        os.path.join(args.output_dir, 'model_state_flow_length_{}.pt'.format(args.flow_length)))

            # plot and save results
            with torch.no_grad():
                plot_flow(base_dist, flow, os.path.join(args.output_dir, 'approximating_flow_step{}.png'.format(i)), args)

# --------------------
# Plotting
# --------------------

def plot_flow(base_dist, flow, filename, args):
    n = 200
    lim = 4

    fig, axs = plt.subplots(2, 2, subplot_kw={'aspect': 'equal'})

    # plot target density we're trying to approx
    plot_target_density(u_z, axs[0,0], lim, n)

    # plot posterior approx density
    plot_flow_density(base_dist, flow, axs[0,1], lim, n)

    # plot flow-transformed base dist sample and histogram
    z = base_dist.sample((10000,))
    zk, _ = flow(z)
    zk = zk.cpu().data.numpy()
    axs[1,0].scatter(zk[:,0], zk[:,1], s=10, alpha=0.4)
    axs[1,1].hist2d(zk[:,0], zk[:,1], bins=lim*50, cmap=plt.cm.jet)

    for ax in plt.gcf().axes:
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_target_density(u_z, ax, range_lim=4, n=200, output_dir=None):
    x = torch.linspace(-range_lim, range_lim, n)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze().to(args.device)

    ax.pcolormesh(xx, yy, torch.exp(-u_z(zz)).view(n,n).data, cmap=plt.cm.jet)

    for ax in plt.gcf().axes:
        ax.set_xlim(-range_lim, range_lim)
        ax.set_ylim(-range_lim, range_lim)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.invert_yaxis()

    if output_dir:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_potential_density.png'))
        plt.close()


def plot_flow_density(base_dist, flow, ax, range_lim=4, n=200, output_dir=None):
    x = torch.linspace(-range_lim, range_lim, n)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze().to(args.device)

    # plot posterior approx density
    zzk, sum_log_abs_det_jacobians = flow(zz)
    log_q0 = base_dist.log_prob(zz)
    log_qk = log_q0 - sum_log_abs_det_jacobians
    qk = log_qk.exp().cpu()
    zzk = zzk.cpu()
    ax.pcolormesh(zzk[:,0].view(n,n).data, zzk[:,1].view(n,n).data, qk.view(n,n).data, cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))

    for ax in plt.gcf().axes:
        ax.set_xlim(-range_lim, range_lim)
        ax.set_ylim(-range_lim, range_lim)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.invert_yaxis()

    if output_dir:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'flow_k{}_density.png'.format(len(flow)-1)))
        plt.close()



# --------------------
# Run
# --------------------

if __name__ == '__main__':

    args = parser.parse_args()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # setup flow
    flow = nn.Sequential(AffineTransform(args.learn_base), *[PlanarTransform() for _ in range(args.flow_length)]).to(args.device)

    # setup target potential to approx
    u_z = vars()[args.target_potential]

    # setup base distribution
    base_dist = D.MultivariateNormal(torch.zeros(2).to(args.device), args.base_sigma * torch.eye(2).to(args.device))

    if args.restore_file:
        # get filename
        filename = os.path.basename(args.restore_file)
        args.flow_length = int(filename.partition('length_')[-1].rpartition('.')[0])
        # reset output dir
        args.output_dir = os.path.dirname(args.restore_file)
        # load state
        state = torch.load(args.restore_file, map_location=args.device)
        # compatibility code;
        # 1/ earlier models did not include step and optimizer checkpoints;
        try:
            flow_state = state['flow_state']
            optimizer_state = state['optimizer_state']
            args.start_step = state['step']
        except KeyError:
            # if state is not a dict, load just the model state
            flow_state = state
            optimizer_state = None
        # 2/ some saved checkpoints may not have a first affine layer
        try:
            flow_state['0.mu']
        except KeyError:
            # if no first affine layer, reload a flow model without one
            flow = nn.Sequential(*[PlanarTransform(args.init_sigma) for _ in range(args.flow_length)])
        flow.load_state_dict(flow_state)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.train:
        optimizer = torch.optim.RMSprop(flow.parameters(), lr=args.lr, momentum=0.9, alpha=0.90, eps=1e-6, weight_decay=args.weight_decay)
        if args.restore_file and optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        args.n_steps = args.start_step + args.n_steps
        optimize_flow(base_dist, flow, u_z, optimizer, args)

    if args.evaluate:
        plot_flow(base_dist, flow, os.path.join(args.output_dir, 'approximating_flow.png'), args)

    if args.plot:
        plot_target_density(u_z, plt.gca(), output_dir=args.output_dir)
        plot_flow_density(base_dist, flow, plt.gca(), output_dir=args.output_dir)
