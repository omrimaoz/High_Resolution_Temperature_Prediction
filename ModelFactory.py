from Models import *
from torch import nn
from IRMaker import IRMaker

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def create_model(model_name, train_dl, valid_dl, means, inputs_dim, loss_weights, criterion=None, opt=None):
        loss_weights = loss_weights if loss_weights is not None else np.ones(TEMP_SCALE * IR_TEMP_FACTOR)
        if not criterion or isinstance(criterion, types.FunctionType) or criterion == nn.MSELoss:
            outputs_dim = 1
        else:
            # CrossEntropyLoss
            outputs_dim = TEMP_SCALE * IR_TEMP_FACTOR
            criterion = criterion(weight=torch.tensor(loss_weights, dtype=torch.float))

        if model_name == 'IRValue':
            if criterion:
                return IRValue(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion, opt)
            return IRValue(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion=nn.MSELoss(), opt=None)
        if model_name == 'IRClass':
            if criterion:
                return IRClass(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion, opt)
            return IRClass(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion=nn.CrossEntropyLoss(), opt=None)
        if model_name == 'ConvNet':
            if criterion:
                return ConvNet(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion, opt)
            return ConvNet(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion=nn.MSELoss(), opt=None)
        if model_name == 'ResNet18':
            return ResNet18(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion=nn.CrossEntropyLoss(), opt=None)
        if model_name == 'ResNet50':
            return ResNet50(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion=nn.CrossEntropyLoss(), opt=None)
        if model_name == 'InceptionV3':
            return InceptionV3(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion=nn.CrossEntropyLoss(), opt=None)
        if model_name == 'VGG19':
            return VGG19(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion=nn.CrossEntropyLoss(), opt=None)
        if model_name == 'ResNetXt101':
            return ResNetXt101(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion=nn.CrossEntropyLoss(), opt=None)

        return None
