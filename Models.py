from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from Loss_Functions import *
from utils import *
from IRMaker import IRMaker
import types


class TemperatureModel(torch.nn.Module):
    cache = {
        'accuracy': list(),
        'MAE': list(),
        'MSE': list(),
        'train_prediction': None,
        'actual_mean': 0.
    }

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.means = means
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        self.criterion = criterion
        self.opt = opt

    def find_close_means(self, target, device):
        matrix = torch.tile(torch.tensor(self.means), dims=(target.shape[0], 1)).to(device)
        close_means = torch.abs(matrix.T - target).T
        close_means_indices = torch.argmin(close_means, dim=1)
        mask = torch.zeros_like(matrix)
        for i, idx in enumerate(close_means_indices):
            mask[i][idx] = 1
        return matrix * mask

    def model_loss(self, output, target, device):
        if isinstance(self.criterion, types.FunctionType):
            if self.criterion.__name__ == 'WMSELoss':
                means = self.find_close_means(target, device)
                output = output.view(-1)
                return self.criterion(output, target, means, device)
            elif self.criterion.__name__ == 'CVLoss':
                const = 100
                output = output.view(-1)
                return self.criterion(output, target, const, device)

        return self.criterion(output, target)

    def predict(self, y_hat):
        if self.outputs_dim > 1:
            return torch.max(y_hat, 1)[1]
        else:
            return y_hat.detach().view(-1)


class IRValue(TemperatureModel):
    name = 'IRValue'
    epochs = 50
    lr = 1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt):
        super(IRValue, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt)
        self.linear1 = nn.Linear(inputs_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.fc = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def unpack(self, pack, device, opt):
        return pack[0].float().to(device), pack[1].float().to(device)

    def lambda_scheduler(self, epoch):
        if epoch < 15:
            return 0.1
        if epoch < 30:
            return 0.005
        return 0.0001


class IRClass(TemperatureModel):
    name = 'IRClass'
    epochs = 300
    lr = 1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt):
        super(IRClass, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt)
        self.linear1 = nn.Linear(inputs_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, self.outputs_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def unpack(self, pack, device, opt):
        return pack[0].float().to(device), torch.tensor(pack[1]).long().to(device)

    def lambda_scheduler(self, epoch):
        if epoch < 50:
            return 0.01
        if epoch < 100:
            return 0.001
        return 0.0005


class FTP(TemperatureModel):
    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion=nn.CrossEntropyLoss(), opt=None):
        super(FTP, self).__init__(train_loader, valid_loader, means, IRMaker.DATA_MAPS_COUNT + IRMaker.STATION_PARAMS_COUNT  - opt['pretrained_ResNet18_correction'], outputs_dim, criterion, opt)

    def unpack(self, pack, device, opt):
        X, y = pack
        if X.shape[1] == IRMaker.FRAME_WINDOW**2 * (IRMaker.DATA_MAPS_COUNT - opt['pretrained_ResNet18_correction']) + IRMaker.STATION_PARAMS_COUNT:
            y = y.long().to(device) if self.outputs_dim > 1 else y.float().to(device)
            X, data = X.float()[:, :IRMaker.FRAME_WINDOW**2 * (IRMaker.DATA_MAPS_COUNT - opt['pretrained_ResNet18_correction'])].to(device),\
                        X.float()[:, IRMaker.FRAME_WINDOW**2 * (IRMaker.DATA_MAPS_COUNT - opt['pretrained_ResNet18_correction']):].to(device)
            X = X.reshape((X.shape[0], IRMaker.DATA_MAPS_COUNT - opt['pretrained_ResNet18_correction'],
                           IRMaker.FRAME_WINDOW, IRMaker.FRAME_WINDOW))
        else:
            y = y.long().to(device) if self.outputs_dim > 1 else y.float().to(device)
            X, data = X.float()[:, :IRMaker.FRAME_WINDOW ** 2 * 3].to(device), \
                        X.float()[:, IRMaker.FRAME_WINDOW ** 2 * 3:].to(device)
            X = X.reshape((X.shape[0], 3, IRMaker.FRAME_WINDOW, IRMaker.FRAME_WINDOW))

        return X, y, data


class ConvNet(FTP):
    name = 'ConvNet'
    epochs = 20000
    lr = 1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion=nn.CrossEntropyLoss(), opt=None):
        super(ConvNet, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt)
        self.kernel_size = 5
        self.conv_output = ((((IRMaker.FRAME_WINDOW - (self.kernel_size - 1)) // 2) - (self.kernel_size - 1)) // 2)
        in_channels = (inputs_dim - IRMaker.STATION_PARAMS_COUNT) // (IRMaker.FRAME_WINDOW ** 2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 6, kernel_size=self.kernel_size)
        torch.nn.init.xavier_uniform(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.conv2 = nn.Conv2d(in_channels=in_channels * 6, out_channels=64, kernel_size=self.kernel_size)
        self.fc1 = nn.Linear(64 * self.conv_output ** 2 + IRMaker.STATION_PARAMS_COUNT, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.outputs_dim)

    def forward(self, x, data=torch.Tensor()):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = torch.flatten(x, 1)
        x2 = data
        x = torch.cat((x, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # CE
    def lambda_scheduler(self, epoch):
        # CE
        if self.opt['isCE']:
            if epoch < 35:
                return 0.001
            if epoch < 150:
                return 0.0005
            return 0.0001
        # WMSE, MSE
        else:
            # if epoch < 50:
            #     return 0.001
            # if epoch < 500:
            #     return 0.0001
            return 0.00001


class DeeperConvNet(FTP):
    name = 'DeeperConvNet'
    epochs = 20000
    lr = 1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion=nn.CrossEntropyLoss(), opt=None):
        super(DeeperConvNet, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt)
        self.kernel_size = 4
        self.pad = 1
        self.conv_output = IRMaker.FRAME_WINDOW
        for i in range(4):
            if i % 2 == 0:
                self.conv_output = (self.conv_output - (self.kernel_size - 1) + self.pad * 2) // 2
            else:
                self.conv_output = (self.conv_output - (self.kernel_size - 1) + self.pad * 2)
        in_channels = (inputs_dim - IRMaker.STATION_PARAMS_COUNT) // (IRMaker.FRAME_WINDOW ** 2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 6, kernel_size=self.kernel_size, padding=self.pad)
        torch.nn.init.xavier_uniform(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.bn1 = nn.BatchNorm2d(in_channels * 6)
        self.conv2 = nn.Conv2d(in_channels=in_channels * 6, out_channels=in_channels * 36, kernel_size=self.kernel_size, padding=self.pad)
        self.conv3 = nn.Conv2d(in_channels=in_channels * 36, out_channels=in_channels * 72, kernel_size=self.kernel_size, padding=self.pad)
        self.bn3 = nn.BatchNorm2d(in_channels * 72)
        self.conv4 = nn.Conv2d(in_channels=in_channels * 72, out_channels=64, kernel_size=self.kernel_size, padding=self.pad)

        self.fc = nn.Linear(64 * self.conv_output ** 2 + IRMaker.STATION_PARAMS_COUNT, self.outputs_dim)

    def forward(self, x, data=torch.Tensor()):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.relu(self.conv4(x))

        x = torch.flatten(x, 1)
        x2 = data
        x = torch.cat((x, x2), dim=1)
        x = F.relu(self.fc(x))
        return x

    # CE
    def lambda_scheduler(self, epoch):
        # CE
        if self.opt['isCE']:
            if epoch < 35:
                return 0.01
            if epoch < 150:
                return 0.05
            return 0.001
        # WMSE, MSE
        else:
            # if epoch < 50:
            #     return 0.001
            # if epoch < 500:
            #     return 0.0001
            return 0.00001


class PretrainedModel(FTP):
    epochs = 100
    lr = 0.1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt, pretrained_model):
        super(PretrainedModel, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt)
        self.pretrained_model = pretrained_model
        self.fc_inputs = IRMaker.STATION_PARAMS_COUNT
        self.fc_with_data = None

    def forward(self, x, data=torch.Tensor()):
        x = self.pretrained_model(x)
        x = torch.cat((x, data), dim=1)
        x = self.fc_with_data(x)
        return x

    def lambda_scheduler(self, epoch):
        if epoch < 30:
            return 0.1
        if epoch < 60:
            return 0.01
        return 0.0005


class ResNet18(PretrainedModel):
    name = 'ResNet18'
    epochs = 3000
    lr = 0.1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt):
        if opt['use_pretrained_weights']:
            super(ResNet18, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt, models.resnet18(pretrained=True))
        else:
            super(ResNet18, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt, models.resnet18(pretrained=False))
            self.pretrained_model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            torch.nn.init.xavier_uniform(self.pretrained_model.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.fc_inputs += 1000
        self.fc_with_data = nn.Sequential(
            nn.Linear(self.fc_inputs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.outputs_dim))

    def lambda_scheduler(self, epoch):
        if epoch < 30:
            return 0.005
        if epoch < 60:
            return 0.0005
        if epoch < 100:
            return 0.00005
        return 0.00001


class ResNet50(PretrainedModel):
    name = 'ResNet50'
    epochs = 200
    lr = 1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt):
        super(ResNet50, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt, models.resnet50(pretrained=False))
        self.pretrained_model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        torch.nn.init.xavier_uniform(self.pretrained_model.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.fc_inputs += 1000
        self.fc_with_data = nn.Sequential(
            nn.Linear(self.fc_inputs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.outputs_dim))

    def lambda_scheduler(self, epoch):
        if epoch < 20:
            return 0.1
        if epoch < 60:
            return 0.01
        return 0.001

class InceptionV3(PretrainedModel):
    name = 'InceptionV3'
    epochs = 100
    lr = 1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt):
        super(InceptionV3, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt, models.inception_v3(pretrained=False))
        self.pretrained_model.Conv2d_1a_3x3.conv = nn.Conv2d(6, 32, kernel_size=3, stride=2, bias=False)
        torch.nn.init.xavier_uniform(self.pretrained_model.Conv2d_1a_3x3.conv.weight, gain=nn.init.calculate_gain('relu'))
        self.fc_inputs += 1000
        self.fc_with_data = nn.Sequential(
            nn.Linear(self.fc_inputs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.outputs_dim))
        self.pretrained_model.aux_logits = False

    # def unpack(self, pack, device):
    #     X, y, data = pack[0].float()[:, :299**2 * IRMaker.DATA_MAPS_COUNT].to(device),\
    #                  torch.round(pack[1]).long().to(device), pack[0].float()[:, 299**2 * IRMaker.DATA_MAPS_COUNT:].to(device)
    #     X = X.reshape((pack[0].size()[0], IRMaker.DATA_MAPS_COUNT, 299, 299))  # TODO replace with params
    #     return X, y, data


class VGG19(PretrainedModel):
    name = 'VGG19'
    epochs = 100
    lr = 1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt):
        super(VGG19, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt, models.vgg19(pretrained=False))
        self.pretrained_model.features[0] = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.xavier_uniform(self.pretrained_model.features[0].weight, gain=nn.init.calculate_gain('relu'))
        self.fc_inputs += 1000
        self.fc_with_data = nn.Sequential(
            nn.Linear(self.fc_inputs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.outputs_dim))

    # def unpack(self, pack, device):
    #     X, y, data = pack[0].float()[:, :224**2 * IRMaker.DATA_MAPS_COUNT].to(device),\
    #                  torch.round(pack[1]).long().to(device), pack[0].float()[:, 224**2 * IRMaker.DATA_MAPS_COUNT:].to(device)
    #     X = X.reshape((pack[0].size()[0], IRMaker.DATA_MAPS_COUNT, 224, 224))  # TODO replace with params
    #     return X, y, data


class ResNetXt101(PretrainedModel):
    name = 'ResNetXt101'
    epochs = 1000
    lr = 1

    def __init__(self, train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt):
        super(ResNetXt101, self).__init__(train_loader, valid_loader, means, inputs_dim, outputs_dim, criterion, opt, models.resnet101(pretrained=False))
        self.pretrained_model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        torch.nn.init.xavier_uniform(self.pretrained_model.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.fc_inputs += 1000
        self.fc_with_data = nn.Sequential(
            nn.Linear(self.fc_inputs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.outputs_dim))

    def lambda_scheduler(self, epoch):
        if epoch < 20:
            return 0.1
        if epoch < 200:
            return 0.01
        return 0.001
