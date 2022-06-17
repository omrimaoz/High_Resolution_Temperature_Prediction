from Models import IRValue, IRClass, ConvNet


class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def create_model(model_name, train_dl, valid_dl, inputs_dim, images_dim=1):
        if model_name == 'IRValue':
            return IRValue(train_dl, valid_dl, inputs_dim)
        if model_name == 'IRClass':
            return IRClass(train_dl, valid_dl, inputs_dim)
        if model_name == 'ConvNet':
            return ConvNet(train_dl, valid_dl, inputs_dim, images_dim=images_dim)
        return None
