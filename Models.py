import torch
from torch import nn

from utils import IR_TEMP_FACTOR


class TemperatureModel(torch.nn.Module):
    cache = {
        'accuracy': list(),
        'MAE': list(),
        'MSE': list(),
        'train_prediction': None,
        'actual_mean': 0.
    }

    def __init__(self, train_loader, valid_loader, inputs_dim, outputs_dim, criterion):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        self.criterion = criterion


class IRValue(TemperatureModel):
    name = 'IRValue'
    epochs = 50
    lr = 1

    def __init__(self, train_loader, valid_loader, inputs_dim, outputs_dim=1, criterion=nn.MSELoss()):
        super(IRValue, self).__init__(train_loader, valid_loader, inputs_dim, outputs_dim, criterion)
        self.linear1 = nn.Linear(inputs_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        # self.linear3 = nn.Linear(128, 256)
        # self.linear4 = nn.Linear(128, 256)
        self.fc = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        # x = self.relu(self.linear3(x))
        # x = self.relu(self.linear4(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def unpack(self, pack):
        return pack[0].float(), pack[1].float()

    def predict(self, y_hat):
        return y_hat.detach().view(-1)

    def lambda_scheduler(self, epoch):
        return 0.1
        if epoch < 15:
            return 0.1
        if epoch < 30:
            return 0.005
        return 0.0001


class IRClass(TemperatureModel):
    name = 'IRClass'
    epochs = 20
    lr = 1

    def __init__(self, train_loader, valid_loader, inputs_dim, outputs_dim=70 * IR_TEMP_FACTOR, criterion=nn.CrossEntropyLoss()):
        super(IRClass, self).__init__(train_loader, valid_loader, inputs_dim, outputs_dim, criterion)
        self.linear1 = nn.Linear(inputs_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, outputs_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
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
        if epoch < 50:
            return 0.001
        return 0.0005

