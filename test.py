import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader, Dataset

import unittest
from unittest.mock import MagicMock

from maf import MADE, MADEMOG, MAF, MAFMOG, RealNVP, BatchNorm, LinearMaskedCoupling, train
from glow import Actnorm, Invertible1x1Conv, AffineCoupling, Squeeze, Split, FlowStep, FlowLevel, Glow, train_epoch
from data import fetch_dataloaders


args = MagicMock()
args.input_size = 1000
args.batch_size = 100
args.device = torch.device('cpu')

NORM_TOL = 1e-4  # tolerance for difference in vector norms

torch.manual_seed(1)

# --------------------
# Test invertibility and log dets of individual layers
# --------------------

def test_layer(l, input_dims, cond_label_size=None, norm_tol=NORM_TOL):
    x = torch.randn(input_dims)
    batch_size = input_dims[0]

    labels = None
    if cond_label_size is not None:  # make one hot labels
        labels = torch.eye(cond_label_size).repeat(batch_size // cond_label_size + 1, 1)[:batch_size]

    u, logd = l(x) if labels is None else l(x, labels)
    recon_x, inv_logd = l.inverse(u) if labels is None else l.inverse(u, labels)

    d_data, d_logd = (recon_x - x).norm(), (logd + inv_logd).norm()
    assert d_data < norm_tol, 'Data reconstruction fail - norm of difference = {}.'.format(d_data)
    assert d_logd < norm_tol, 'Log determinant inversion fail. - norm of difference = {}'.format(d_logd)


class TestFlowLayers(unittest.TestCase):
    def test_batch_norm(self):
        l = BatchNorm(args.input_size)
        l.train()
        for i in range(3):
            with self.subTest(train_loop_iter=i):
                test_layer(l, input_dims=(args.batch_size, args.input_size))

        l.eval()
        for i in range(2):
            with self.subTest(eval_loop_iter=i):
                test_layer(l, input_dims=(args.batch_size, args.input_size))

    def test_linear_coupling(self):
        mask = torch.arange(args.input_size).float() % 2
        # unconditional
        test_layer(LinearMaskedCoupling(args.input_size, hidden_size=10, n_hidden=1, mask=mask), input_dims=(args.batch_size, args.input_size))
        test_layer(LinearMaskedCoupling(args.input_size, hidden_size=10, n_hidden=2, mask=mask), input_dims=(args.batch_size, args.input_size))
        # conditional
        cond_label_size = 10
        test_layer(LinearMaskedCoupling(args.input_size, hidden_size=10, n_hidden=1, mask=mask, cond_label_size=cond_label_size),
                   input_dims=(args.batch_size, args.input_size), cond_label_size=cond_label_size)
        test_layer(LinearMaskedCoupling(args.input_size, hidden_size=10, n_hidden=2, mask=mask, cond_label_size=cond_label_size),
                   input_dims=(args.batch_size, args.input_size), cond_label_size=cond_label_size)

    def test_made(self):
        test_layer(MADE(args.input_size, hidden_size=10, n_hidden=3), input_dims=(args.batch_size, args.input_size))

    def test_actnorm(self):
        test_layer(Actnorm(param_dim=(1,3,1,1)), input_dims=(args.batch_size, 3, 50, 50))

    def test_invertible1x1conv(self):
        test_layer(Invertible1x1Conv(n_channels=24), input_dims=(args.batch_size, 24, 50, 50), norm_tol=1e-3)
        test_layer(Invertible1x1Conv(n_channels=12, lu_factorize=True), input_dims=(args.batch_size, 12, 50, 50), norm_tol=1e-3)

    def test_affinecoupling(self):
        test_layer(AffineCoupling(n_channels=4, width=12), input_dims=(args.batch_size, 4, 50, 50), norm_tol=5e-4)

    def test_squeeze(self):
        net = Squeeze()
        x = torch.rand(args.batch_size, 12, 20, 30)
        recon_x = net.inverse(net(x))
        y = net(net.inverse(x))
        assert torch.allclose(x, recon_x), 'Data reconstruction failed.'
        assert torch.allclose(x, y)

    def test_split(self):
        net = Split(n_channels=10)
        x = torch.randn(args.batch_size, 10, 20, 30)
        x1, z2, logd = net(x)
        recon_x, inv_logd = net.inverse(x1, z2)
        d_data, d_logd = (recon_x - x).norm(), (logd + inv_logd).norm()
        assert d_data < 1e-4, 'Data reconstruction fail - norm of difference = {}.'.format(d_data)
        assert d_logd < 1e-4, 'Log determinant inversion fail. - norm of difference = {}'.format(d_logd)

    def test_flowstep(self):
        test_layer(FlowStep(n_channels=4, width=12), input_dims=(args.batch_size, 4, 50, 50), norm_tol=1e-3)

    def test_flowlevel(self):
        net = FlowLevel(n_channels=3, width=12, depth=2)
        x = torch.randn(args.batch_size, 3, 32, 32)
        x1, z2, logd = net(x)
        recon_x, inv_logd = net.inverse(x1, z2)
        d_data, d_logd = (recon_x - x).norm(), (logd + inv_logd).norm()
        assert d_data < 5e-4, 'Data reconstruction fail - norm of difference = {}.'.format(d_data)
        assert d_logd < 5e-4, 'Log determinant inversion fail. - norm of difference = {}'.format(d_logd)

    def test_glow(self):
        net = Glow(width=12, depth=3, n_levels=3)
        x = torch.randn(args.batch_size, 3, 32, 32)
        zs, logd = net(x)
        recon_x, inv_logd = net.inverse(zs)
        y, _ = net.inverse(batch_size=args.batch_size)
        d_data, d_data_y, d_logd = (recon_x - x).norm(), (x - y).norm(), (logd + inv_logd).norm()
        assert d_data < 1e-3, 'Data reconstruction fail - norm of difference = {}.'.format(d_data)
#        assert d_data_y < 1e-3, 'Data reconstruction (inv > base > inv) fail - norm of difference = {}.'.format(d_data_y)
        assert d_logd < 1e-3, 'Log determinant inversion fail. - norm of difference = {}'.format(d_logd)

# --------------------
# Test MAF
# --------------------

# Test flow invertibility (KL=0) at initalization
@torch.no_grad()
def test_untrained_model(model, cond_label_size=None):
    # 1. sample Gaussian data;
    # 2. run model forward and reverse;
    # 3. roconstruct data; 
    # 4. measure KL between Gaussian fitted to the data and the base distribution
    n_samples = 1000
    data = model.base_dist.sample((n_samples,))
    labels = None
    if cond_label_size is not None:  # make one hot labels
        labels = torch.eye(cond_label_size).repeat(n_samples // cond_label_size + 1, 1)[:n_samples]
    u, logd = model(data, labels)
    recon_data, _ = model.inverse(u, labels)
    recon_dist = D.Normal(recon_data.mean(0), recon_data.var(0).sqrt())
    kl = D.kl.kl_divergence(recon_dist, model.base_dist).sum(-1)
    print('KL (q || p) = {:.4f}'.format(kl))

# Test flow can train to random numbers in N(0,1) ie KL(random numbers driving flow || base distribution) = 0
@torch.no_grad()
def test_trained_model(model, dl, cond_label_size=None):
    # 1. sample toy data;
    # 2. run model forward and generate random numbers driving the model;
    # 3. measure KL between Gaussian fitted to random numbers driving the model and the base distribution
    data, labels = next(iter(dl))
    labels = None
    if cond_label_size is not None:  # make one hot labels
        labels = torch.eye(cond_label_size).repeat(n_samples // cond_label_size + 1, 1)[:n_samples]
    u, logd = model(data, labels)
    u_dist = D.Normal(u.mean(0), u.std(0))
    kl = D.kl.kl_divergence(u_dist, model.base_dist).sum()
    print('KL (u || p) = {:.4f}'.format(kl))

class TestMAFUntrained(unittest.TestCase):
    def setUp(self):
        self.cond_label_size = 2

    def test_made_1_hidden(self):
        test_untrained_model(MADE(input_size=2, hidden_size=10, n_hidden=1, cond_label_size=None, activation='relu', input_order='sequential'))

    def test_made_1_hidden_conditional(self):
        test_untrained_model(MADE(input_size=2, hidden_size=10, n_hidden=1, cond_label_size=self.cond_label_size, activation='relu',
            input_order='sequential'), self.cond_label_size)

    def test_made_2_hidden(self):
        test_untrained_model(MADE(input_size=2, hidden_size=10, n_hidden=2, cond_label_size=None, activation='relu', input_order='sequential'))

    def test_made_2_hidden_conditional(self):
        test_untrained_model(MADE(input_size=2, hidden_size=10, n_hidden=2, cond_label_size=self.cond_label_size, activation='relu',
            input_order='sequential'), self.cond_label_size)

    def test_made_200_inputs_random_mask(self):
        test_untrained_model(MADE(input_size=200, hidden_size=10, n_hidden=2, cond_label_size=None, activation='relu', input_order='random'))

    def test_maf_1_blocks_no_bn(self):
        test_untrained_model(MAF(n_blocks=1, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=None, 
                           activation='relu', input_order='sequential', batch_norm=False))

    def test_maf_1_blocks_bn(self):
        test_untrained_model(MAF(n_blocks=1, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=None, 
                           activation='relu', input_order='sequential', batch_norm=True))

    def test_maf_2_blocks(self):
        test_untrained_model(MAF(n_blocks=2, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=None, 
                           activation='relu', input_order='sequential', batch_norm=True))

    def test_maf_1_blocks_conditional(self):
        test_untrained_model(MAF(n_blocks=1, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=self.cond_label_size, 
                           activation='relu', input_order='sequential', batch_norm=True), self.cond_label_size)

    def test_maf_2_blocks_conditional(self):
        test_untrained_model(MAF(n_blocks=2, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=self.cond_label_size, 
                           activation='relu', input_order='sequential', batch_norm=True), self.cond_label_size)

    def test_realnvp_1_block_200_inputs(self):
        test_untrained_model(RealNVP(n_blocks=1, input_size=200, hidden_size=10, n_hidden=2, cond_label_size=None))

    def test_realnvp_2_block_200_inputs(self):
        test_untrained_model(RealNVP(n_blocks=2, input_size=200, hidden_size=10, n_hidden=2, cond_label_size=None))

    def test_realnvp_2_blocks_conditional(self):
        test_untrained_model(RealNVP(n_blocks=2, input_size=200, hidden_size=10, n_hidden=2, cond_label_size=self.cond_label_size),
                self.cond_label_size)

    def test_mademog_1_comp(self):
        test_untrained_model(MADEMOG(n_components=1, input_size=10, hidden_size=10, n_hidden=2, cond_label_size=None, activation='relu',
            input_order='sequential'))

    def test_mademog_10_comp(self):
        test_untrained_model(MADEMOG(n_components=10, input_size=200, hidden_size=10, n_hidden=2, cond_label_size=None, activation='relu',
            input_order='sequential'))

    def test_mafmog_1_block_1_comp(self):
        test_untrained_model(MAFMOG(n_blocks=1, n_components=1, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=self.cond_label_size,
            activation='relu', input_order='sequential', batch_norm=True), self.cond_label_size)

    def test_mafmog_2_blocks_10_comp_conditional(self):
        test_untrained_model(MAFMOG(n_blocks=2, n_components=10, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=self.cond_label_size,
                           activation='relu', input_order='sequential', batch_norm=True), self.cond_label_size)

class TestMAFTrained(unittest.TestCase):
    def setUp(self):
        args = MagicMock()
        args.cond_label_size = None
        args.batch_size = 100
        args.device = torch.device('cpu')
        dl, _ = fetch_dataloaders('TOY', args.batch_size, args.device, flip_toy_var_order=False)

        def _train(model, n_steps):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
            print('Untrained: ')
            test_trained_model(model, dl)
            for _ in range(n_steps):
                train(model, dl, optimizer, 0, args)
            print('Trained: ')
            test_trained_model(model, dl)

        self._train = _train

    def test_made(self):
        model = MADE(input_size=2, hidden_size=100, n_hidden=1, cond_label_size=None, activation='relu', input_order='sequential')
        self._train(model, 10)

    def test_mademog(self):
        model = MADEMOG(n_components=10, input_size=2, hidden_size=100, n_hidden=1, cond_label_size=None, activation='relu', input_order='sequential')
        self._train(model, 5)

    def test_maf_5(self):
        model = MAF(n_blocks=5, input_size=2, hidden_size=100, n_hidden=1, cond_label_size=None,
                           activation='relu', input_order='sequential', batch_norm=True)
        self._train(model, 1)

    def test_mafmog_5_comp_1(self):
        model = MAFMOG(n_blocks=5, n_components=1, input_size=2, hidden_size=100, n_hidden=1, 
                    cond_label_size=None, activation='relu', input_order='sequential', batch_norm=True)
        self._train(model, 5)

    def test_mafmog_5_comp_10(self):
        model = MAFMOG(n_blocks=5, n_components=10, input_size=2, hidden_size=100, n_hidden=1,
                    cond_label_size=None, activation='relu', input_order='sequential', batch_norm=True)
        self._train(model, 10)

    def test_train_realnvp_5(self):
        model = RealNVP(n_blocks=5, input_size=2, hidden_size=100, n_hidden=1, cond_label_size=None, batch_norm=True)
        self._train(model, 5)


# --------------------
# Test Glow
# --------------------

# Generate a dataset from a 2-dim Gaussian distribution and expand to `image size` of (3,32,32)

class ToyDistribution(D.Distribution):
    def __init__(self, flip_var_order):
        super().__init__()
        self.flip_var_order = flip_var_order
        self.p_x2 = D.Normal(0, 4)
        self.p_x1 = lambda x2: D.Normal(0.25 * x2**2, 1)

    def rsample(self, sample_shape=torch.Size()):
        x2 = self.p_x2.sample(sample_shape)
        x1 = self.p_x1(x2).sample()
        if self.flip_var_order:
            return torch.stack((x2, x1), dim=-1).expand(3,-1,-1)
        else:
            return torch.stack((x1, x2), dim=0).repeat(16,1).expand(3,-1,-1)

    def log_prob(self, value):
        if self.flip_var_order:
            value = value.flip(1)
        return self.p_x1(value[:,1]).log_prob(value[:,0]) + self.p_x2.log_prob(value[:,1])


class TOY(Dataset):
    def __init__(self, dataset_size=2500, flip_var_order=False):
        self.input_size = 32
        self.label_size = 1
        self.dataset_size = dataset_size
        self.base_dist = ToyDistribution(flip_var_order)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.base_dist.sample((32,)), torch.zeros(self.label_size)


class TestGlowUntrained(unittest.TestCase):
    def setUp(self):
        def test_kl(model):
            n_samples = 1000
            data = model.base_dist.sample((n_samples,3,32,32)).squeeze()
            zs, logd = model(data)
            recon_data, _ = model.inverse(zs)
            recon_dist = D.Normal(recon_data.mean(0), recon_data.var(0).sqrt())
            kl = D.kl.kl_divergence(recon_dist, model.base_dist).mean()
            print('Model: depth {}, levels {}; Avg per pixel KL (q||p) = {:.4f}'.format(
                len(model.flowstep), len(model.flowlevels), kl))
        self.test_kl = test_kl

    def test_glow_depth_1_levels_1(self):
        # 1. sample data; 2. run model forward and reverse; 3. roconstruct data; 4. measure KL between Gaussian fitted to the data and the base distribution
        self.test_kl(Glow(width=12, depth=1, n_levels=1))

    def test_glow_depth_2_levels_2(self):
        # 1. sample data; 2. run model forward and reverse; 3. roconstruct data; 4. measure KL between Gaussian fitted to the data and the base distribution
        self.test_kl(Glow(width=12, depth=2, n_levels=2))


class TestGlowTrained(unittest.TestCase):
    def setUp(self):
        args = MagicMock()
        args.device = torch.device('cpu')
        dl = DataLoader(TOY(), batch_size=100)

        @torch.no_grad()
        def _test_trained_model(model, dl, cond_label_size=None):
            data, _ = next(iter(dl))
            zs, logd = model(data)
            zs = torch.cat([z.flatten(1) for z in zs], dim=1)  # flatten the z's and concat
            zs_dist = D.Normal(zs.mean(0), zs.std(0))
            kl = D.kl.kl_divergence(zs_dist, model.base_dist).mean()
            print('Mean per pixel KL (zs || N(0,1)) = {:.4f}'.format(kl))
            print('Mean data bits per pixel: {:.4f}'.format(-model.log_prob(data, bits_per_pixel=True).mean(0)))

        def _train(model, n_steps):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
            print('Model: depth {}, levels {}. Untrained:'.format(len(model.flowstep), len(model.flowlevels)))
            _test_trained_model(model, dl)
            for _ in range(n_steps):
                train_epoch(model, dl, optimizer, 0, args)
            print('Trained: ')
            _test_trained_model(model, dl)

        self._train = _train

    def test_glow_1_1(self):
        model = Glow(width=12, depth=1, n_levels=1)
        self._train(model, 3)

    def test_glow_3_3(self):
        model = Glow(width=24, depth=3, n_levels=3)
        self._train(model, 3)


if __name__ == '__main__':
    unittest.main()
