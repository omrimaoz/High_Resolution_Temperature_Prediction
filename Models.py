from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

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


class ConvNet(TemperatureModel):
    name = 'ConvNet'
    epochs = 100
    lr = 1

    def __init__(self, train_loader, valid_loader, inputs_dim, outputs_dim=70 * IR_TEMP_FACTOR, criterion=nn.CrossEntropyLoss(), images_dim=1):
        super(ConvNet, self).__init__(train_loader, valid_loader, images_dim + 4, outputs_dim, criterion)
        self.conv1 = nn.Conv2d(in_channels=images_dim, out_channels=images_dim * 6, kernel_size=5)
        # self.bn1 = nn.BatchNorm2d(images_dim * 6)
        self.conv2 = nn.Conv2d(in_channels=images_dim * 6, out_channels=64, kernel_size=5)
        # self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 9 + 4, 120)  # TODO why 9?
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, outputs_dim)

    def forward(self, x, data=torch.Tensor()):
        # x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        # x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = torch.flatten(x, 1)
        x2 = data
        x = torch.cat((x, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def unpack(self, pack):
        X, y, data = pack[0].float()[:, :3750], torch.round(pack[1]).long(), pack[0].float()[:, 3750:]
        X = X.reshape((pack[0].size()[0], 6, 25, 25))  # TODO replace with params
        return X, y, data

    def predict(self, y_hat):
        return torch.max(y_hat, 1)[1]

    def lambda_scheduler(self, epoch):
        if epoch < 40:
            return 0.01
        if epoch < 75:
            return 0.001
        return 0.0005

# class UNet(nn.Module):
#
#     def __init__(self, in_channels=3, out_channels=1, init_features=32):
#         super(UNet, self).__init__()
#
#         features = init_features
#         self.encoder1 = UNet._block(in_channels, features, name="enc1")
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder2 = UNet._block(features, features * 2, name="enc2")
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
#
#         self.upconv4 = nn.ConvTranspose2d(
#             features * 16, features * 8, kernel_size=2, stride=2
#         )
#         self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
#         self.upconv3 = nn.ConvTranspose2d(
#             features * 8, features * 4, kernel_size=2, stride=2
#         )
#         self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
#         self.upconv2 = nn.ConvTranspose2d(
#             features * 4, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
#         self.upconv1 = nn.ConvTranspose2d(
#             features * 2, features, kernel_size=2, stride=2
#         )
#         self.decoder1 = UNet._block(features * 2, features, name="dec1")
#
#         self.conv = nn.Conv2d(
#             in_channels=features, out_channels=out_channels, kernel_size=1
#         )
#
#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))
#
#         bottleneck = self.bottleneck(self.pool4(enc4))
#
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         return torch.sigmoid(self.conv(dec1))
#
#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv2d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "norm1", nn.BatchNorm2d(num_features=features)),
#                     (name + "relu1", nn.ReLU(inplace=True)),
#                     (
#                         name + "conv2",
#                         nn.Conv2d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "norm2", nn.BatchNorm2d(num_features=features)),
#                     (name + "relu2", nn.ReLU(inplace=True)),
#                 ]
#             )
#         )