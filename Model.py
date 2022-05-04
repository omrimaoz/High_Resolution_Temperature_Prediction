import torch
from torch import nn


class Model(torch.nn.Module):
    def __init__(self, inputs_dim):
        super().__init__()
        self.linear1 = nn.Linear(inputs_dim, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 256)
        self.fc = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x

