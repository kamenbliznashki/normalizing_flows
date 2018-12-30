import torch
import torch.distributions as D
from torch.utils.data import Dataset

from sklearn.datasets import make_moons


class MOONS(Dataset):
    def __init__(self, dataset_size=25000, **kwargs):
        self.x, self.y = make_moons(n_samples=dataset_size, shuffle=True, noise=0.05)
        self.input_size = 2
        self.label_size = 2
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.x[i], self.y[i]




