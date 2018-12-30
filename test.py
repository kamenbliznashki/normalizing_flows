import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import unittest
from unittest.mock import MagicMock

from maf import MADE, MADEMOG, MAF, MAFMOG, RealNVP, BatchNorm, LinearMaskedCoupling, train
from data import fetch_dataloaders


args = MagicMock()
args.input_size = 1000
args.batch_size = 100
args.device = torch.device('cpu')

NORM_TOL = 1e-4  # tolerance for difference in vector norms



# --------------------
# Test invertibility and log dets of individual layers
# --------------------

def test_layer(l, cond_label_size=None):
#    base_dist = D.Normal(0, 1)
#    x = base_dist.sample((args.batch_size, args.input_size))
    x = torch.randn(args.batch_size, args.input_size)

    labels = None
    if cond_label_size is not None:  # make one hot labels
        labels = torch.eye(cond_label_size).repeat(args.batch_size // cond_label_size + 1, 1)[:args.batch_size]

    u, logd = l(x, labels)
    recon_x, inv_logd = l.inverse(u, labels)

    d_data, d_logd = (recon_x - x).norm(), (logd + inv_logd).norm()
    assert d_data < NORM_TOL, 'Data reconstruction fail - norm of difference = {}.'.format(d_data)
    assert d_logd < NORM_TOL, 'Log determinant inversion fail. - norm of difference = {}'.format(d_logd)


class TestFlowLayers(unittest.TestCase):
    def test_batch_norm(self):
        l = BatchNorm(args.input_size)
        l.train()
        for i in range(3):
            with self.subTest(train_loop_iter=i):
                test_layer(l)

        l.eval()
        for i in range(2):
            with self.subTest(eval_loop_iter=i):
                test_layer(l)

    def test_linear_coupling(self):
        mask = torch.arange(args.input_size).float() % 2
        # unconditional
        test_layer(LinearMaskedCoupling(args.input_size, hidden_size=10, n_hidden=1, mask=mask))
        test_layer(LinearMaskedCoupling(args.input_size, hidden_size=10, n_hidden=2, mask=mask))
        # conditional
        cond_label_size = 10
        test_layer(LinearMaskedCoupling(args.input_size, hidden_size=10, n_hidden=1, mask=mask, cond_label_size=cond_label_size), cond_label_size)
        test_layer(LinearMaskedCoupling(args.input_size, hidden_size=10, n_hidden=2, mask=mask, cond_label_size=cond_label_size), cond_label_size)


# --------------------
# Test flows invertibility (KL=0) at initalization
# --------------------

@torch.no_grad()
def test_model(model, cond_label_size=None):
    # 1. sample data; 2. run model forward and reverse; 3. roconstruct data; 4. measure KL between Gaussian fitted to the data and the base distribution
    n_samples = 1000
    data = model.base_dist.sample((n_samples,))
    labels = None
    if cond_label_size is not None:  # make one hot labels
        labels = torch.eye(cond_label_size).repeat(n_samples // cond_label_size + 1, 1)[:n_samples]
    u, logd = model(data, labels)
    recon_data, _ = model.inverse(u, labels)
    recon_dist = D.Normal(recon_data.mean(0), recon_data.var(0).sqrt())
    kl = D.kl.kl_divergence(recon_dist, model.base_dist).sum(-1)
    print('KL (q||p) = {:.4f}'.format(kl))

class TestUntrainedFlows(unittest.TestCase):
    def setUp(self):
        self.cond_label_size = 2

    def test_made_1_hidden(self):
        test_model(MADE(input_size=2, hidden_size=10, n_hidden=1, cond_label_size=None, activation='relu', input_order='sequential'))

    def test_made_1_hidden_conditional(self):
        test_model(MADE(input_size=2, hidden_size=10, n_hidden=1, cond_label_size=self.cond_label_size, activation='relu',
            input_order='sequential'), self.cond_label_size)

    def test_made_2_hidden(self):
        test_model(MADE(input_size=2, hidden_size=10, n_hidden=2, cond_label_size=None, activation='relu', input_order='sequential'))

    def test_made_2_hidden_conditional(self):
        test_model(MADE(input_size=2, hidden_size=10, n_hidden=2, cond_label_size=self.cond_label_size, activation='relu',
            input_order='sequential'), self.cond_label_size)

    def test_made_200_inputs_random_mask(self):
        test_model(MADE(input_size=200, hidden_size=10, n_hidden=2, cond_label_size=None, activation='relu', input_order='random'))

    def test_maf_1_blocks_no_bn(self):
        test_model(MAF(n_blocks=1, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=None, 
                           activation='relu', input_order='sequential', batch_norm=False))

    def test_maf_1_blocks_bn(self):
        test_model(MAF(n_blocks=1, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=None, 
                           activation='relu', input_order='sequential', batch_norm=True))

    def test_maf_2_blocks(self):
        test_model(MAF(n_blocks=2, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=None, 
                           activation='relu', input_order='sequential', batch_norm=True))

    def test_maf_1_blocks_conditional(self):
        test_model(MAF(n_blocks=1, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=self.cond_label_size, 
                           activation='relu', input_order='sequential', batch_norm=True), self.cond_label_size)

    def test_maf_2_blocks_conditional(self):
        test_model(MAF(n_blocks=2, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=self.cond_label_size, 
                           activation='relu', input_order='sequential', batch_norm=True), self.cond_label_size)

    def test_realnvp_1_block_200_inputs(self):
        test_model(RealNVP(n_blocks=1, input_size=200, hidden_size=10, n_hidden=2, cond_label_size=None))

    def test_realnvp_2_block_200_inputs(self):
        test_model(RealNVP(n_blocks=2, input_size=200, hidden_size=10, n_hidden=2, cond_label_size=None))

    def test_realnvp_2_blocks_conditional(self):
        test_model(RealNVP(n_blocks=2, input_size=200, hidden_size=10, n_hidden=2, cond_label_size=self.cond_label_size),
                self.cond_label_size)

    def test_mademog_1_comp(self):
        test_model(MADEMOG(n_components=1, input_size=10, hidden_size=10, n_hidden=2, cond_label_size=None, activation='relu',
            input_order='sequential'))

    def test_mademog_10_comp(self):
        test_model(MADEMOG(n_components=10, input_size=200, hidden_size=10, n_hidden=2, cond_label_size=None, activation='relu',
            input_order='sequential'))

    def test_mafmog_1_block_1_comp(self):
        test_model(MAFMOG(n_blocks=1, n_components=1, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=self.cond_label_size,
            activation='relu', input_order='sequential', batch_norm=True), self.cond_label_size)

    def test_mafmog_2_blocks_10_comp_conditional(self):
        test_model(MAFMOG(n_blocks=2, n_components=10, input_size=2, hidden_size=10, n_hidden=1, cond_label_size=self.cond_label_size,
                           activation='relu', input_order='sequential', batch_norm=True), self.cond_label_size)



# --------------------
# Train a flow to 0 KL
# --------------------

class TestTrainedFlows(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.args.cond_label_size = None
        self.args.batch_size = 100
        self.args.device = torch.device('cpu')
        self.dl, _ = fetch_dataloaders('TOY', self.args.batch_size, self.args.device, flip_toy_var_order=False)

    def test_made(self):
        model = MADE(input_size=2, hidden_size=100, n_hidden=1, cond_label_size=None, activation='relu', input_order='sequential')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        train(model, self.dl, optimizer, 0, self.args)
        test_model(model)

    def test_mademog(self):
        model = MADEMOG(n_components=10, input_size=2, hidden_size=100, n_hidden=1, cond_label_size=None, activation='relu', input_order='sequential')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        for _ in range(1):
            train(model, self.dl, optimizer, 0, self.args)
        test_model(model)

    def test_train_made_flip_var(self):
        self.dl.dataset.flip_toy_var_order = True
        model = MADE(input_size=2, hidden_size=100, n_hidden=1, cond_label_size=None, activation='relu', input_order='sequential')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        train(model, self.dl, optimizer, 0, self.args)
        test_model(model)

    def test_train_maf_5(self):
        model = MAF(n_blocks=5, input_size=2, hidden_size=100, n_hidden=1, cond_label_size=None,
                           activation='relu', input_order='sequential', batch_norm=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        for i in range(2):
            train(model, self.dl, optimizer, i, self.args)
        test_model(model)

    def test_train_mafmog_5_comp_1(self):
        model = MAFMOG(n_blocks=5, n_components=1, input_size=2, hidden_size=100, n_hidden=1, 
                    cond_label_size=None, activation='relu', input_order='sequential', batch_norm=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        for i in range(1):
            train(model, self.dl, optimizer, i, self.args)
        test_model(model)

    def test_train_mafmog_5_comp_10(self):
        model = MAFMOG(n_blocks=5, n_components=10, input_size=2, hidden_size=100, n_hidden=1,
                    cond_label_size=None, activation='relu', input_order='sequential', batch_norm=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        for i in range(1):
            train(model, self.dl, optimizer, i, self.args)
        test_model(model)

    def test_train_realnvp_5(self):
        model = RealNVP(n_blocks=5, input_size=2, hidden_size=100, n_hidden=1, cond_label_size=None, 
                            batch_norm=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        for i in range(2):
            train(model, self.dl, optimizer, i, self.args)
        test_model(model)


if __name__ == '__main__':
    unittest.main()
