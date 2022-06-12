import torch
from torch import nn

from Loss_Functions import *
from utils import IR_TEMP_FACTOR


class TemperatureModel(torch.nn.Module):
    cache = {
        'accuracy': list(),
        'MAE': list(),
        'MSE': list(),
        'train_prediction': None,
        'actual_mean': 0.
    }

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.means = means
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        self.criterion = criterion

    def find_close_means(self, target):
        matrix = torch.tile(torch.tensor(self.means), dims=(target.shape[0], 1))
        close_means = torch.abs(matrix.T - target).T
        close_means_indices = torch.argmin(close_means, dim=1)
        mask = torch.zeros_like(matrix)
        for i, idx in enumerate(close_means_indices):
            mask[i][idx] = 1
        return matrix * mask

    def model_loss(self, output, target):
        if self.criterion.__name__ == 'WMSELoss':
            means = self.find_close_means(target)
            output = output.view(-1)
            return self.criterion(output, target, means)
        elif self.criterion.__name__ == 'CVLoss':
            const = 100
            output = output.view(-1)
            return self.criterion(output, target, const)
        else:
            return self.criterion(output, target)


class IRValue(TemperatureModel):
    name = 'IRValue'
    epochs = 50
    lr = 1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion):
        super(IRValue, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion)
        self.linear1 = nn.Linear(inputs_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        # self.linear3 = nn.Linear(128, 256)
        # self.linear4 = nn.Linear(128, 256)
        self.fc = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        # x = self.relu(self.linear3(x))
        # x = self.relu(self.linear4(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def unpack(self, pack, device):
        return pack[0].float().to(device), pack[1].float().to(device)

    def predict(self, y_hat):
        return y_hat.detach().view(-1)

    def lambda_scheduler(self, epoch):
        if epoch < 15:
            return 0.1
        if epoch < 30:
            return 0.005
        return 0.0001


class IRClass(TemperatureModel):
    name = 'IRClass'
    epochs = 20
    lr = 1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion):
        super(IRClass, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion)
        self.linear1 = nn.Linear(inputs_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, outputs_dim * IR_TEMP_FACTOR)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def unpack(self, pack, device):
        return pack[0].float().to(device), torch.round(pack[1]).long().to(device)

    def predict(self, y_hat):
        return torch.max(y_hat, 1)[1]

    def lambda_scheduler(self, epoch):
        if epoch < 25:
            return 0.01
        if epoch < 50:
            return 0.001
        return 0.0005

