import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.linear = nn.Linear()

    def forward(self):

