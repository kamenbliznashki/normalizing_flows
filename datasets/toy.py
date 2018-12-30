import torch
import torch.distributions as D
from torch.utils.data import Dataset


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
            return torch.stack((x2, x1), dim=-1).squeeze()
        else:
            return torch.stack((x1, x2), dim=-1).squeeze()

    def log_prob(self, value):
        if self.flip_var_order:
            value = value.flip(1)
        return self.p_x1(value[:,1]).log_prob(value[:,0]) + self.p_x2.log_prob(value[:,1])


class TOY(Dataset):
    def __init__(self, dataset_size=25000, flip_var_order=False):
        self.input_size = 2
        self.label_size = 1
        self.dataset_size = dataset_size
        self.base_dist = ToyDistribution(flip_var_order)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.base_dist.sample(), torch.zeros(self.label_size)



