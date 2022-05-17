import torch
from torch import nn

from utils import IR_TEMP_FACTOR


class IRValue(torch.nn.Module):
    name = 'IRValue'
    epochs = 50
    lr = 1

    def __init__(self, inputs_dim):
        super().__init__()
        self.linear1 = nn.Linear(inputs_dim, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, 128)
        # self.linear4 = nn.Linear(128, 256)
        self.fc = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        # x = self.relu(self.linear4(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def unpack(self, pack):
        return pack[0].float(), pack[1].float()

    def predict(self, y_hat):
        return y_hat.detach().view(-1)

    def lambda_scheduler(self, epoch):
        if epoch < 7:
            return 0.1
        if epoch < 15:
            return 0.005
        return 0.0001

class IRClass(torch.nn.Module):
    name = 'IRClass'
    epochs = 35
    lr = 1

    def __init__(self, inputs_dim, outputs_dim=70 * IR_TEMP_FACTOR):
        super().__init__()
        self.linear1 = nn.Linear(inputs_dim, 256)
        self.linear2 = nn.Linear(256, 160)
        self.linear3 = nn.Linear(160, 120)
        self.linear4 = nn.Linear(120, 40)
        self.fc = nn.Linear(40, outputs_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def unpack(self, pack):
        return pack[0].float(), torch.round(pack[1]).long()

    def predict(self, y_hat):
        return torch.max(y_hat, 1)[1]

    def lambda_scheduler(self, epoch):
        if epoch < 25:
            return 0.01
        return 0.001

