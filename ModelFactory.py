from torch import nn
from Models import IRValue, IRClass


class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def create_model(model_name, train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion=None):
        if model_name == 'IRValue':
            if criterion:
                return IRValue(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion)
            return IRValue(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion=nn.MSELoss())
        if model_name == 'IRClass':
            if criterion:
                return IRValue(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion)
            return IRClass(train_dl, valid_dl, means, inputs_dim, outputs_dim, criterion=nn.CrossEntropyLoss())
        return None
