import torch
from torch import nn


class Model(torch.nn.Module):
    def __init__(self, inputs_dim):
        super().__init__(inputs_dim)
        self.linear1 = nn.Linear(inputs_dim, 32)
        self.linear2 = nn.Linear(32, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x

