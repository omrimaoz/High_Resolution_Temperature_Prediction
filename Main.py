import time

import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from datetime import datetime
import torch

from IRMaker import IRMaker
from ModelFactory import ModelFactory
from Prepare_Data import prepare_data
from utils import *
from Dataset import Dataset
from Models import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
print('device: ' + str(device))


def save_model(model, MAE):
    path = '{dir}/{model}_{time}'.format(dir=MODELS_DIR, model=model.name,
                                         time=datetime.now().strftime('%d%m%y'))
    if MAE:
        path += '_mae{mae}'.format(mae=np.round(MAE, ROUND_CONST))
    model.cpu()
    torch.save(model.state_dict(), path + MODEL_EXTENSION)
    model.to(device)
    if model.cache['train_prediction'] is not None:
        model.cache['train_prediction'] = (model.cache['train_prediction'] / IR_TEMP_FACTOR).tolist()
        with open(path + JSON_EXTENSION, 'w') as f:
            f.write(json.dumps(model.cache))
        model.cache['train_prediction'] = np.array(model.cache['train_prediction'])
    print('Model Saved Successfully')

def train_model(model, opt):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=model.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=model.lambda_scheduler)

    MAE = None

    for epoch in range(model.epochs):
        start = time.time()
        model.train()
        sum_loss = 0.0
        total = 0
        for batch, pack in enumerate(model.train_loader):
            x, y, *data = model.unpack(pack, device)
            y_pred = model(x, data[0]).to(device) if data else model(x).to(device)
            y_hat = model.predict(y_pred)

            loss = model.model_loss(y_pred, y, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scale_factor = (TEMP_SCALE * IR_TEMP_FACTOR) ** 2 if opt['normalize'] else 1
            sum_loss += loss.item() * y.shape[0] * scale_factor
            total += y.shape[0]

            if epoch == model.epochs - 1:
                y_cpu, y_hat_cpu = y.cpu(), y_hat.cpu()
                if model.cache['train_prediction'] is not None:
                    model.cache['train_prediction'] = np.hstack((model.cache['train_prediction'],
                                                                      np.vstack([y_cpu, y_hat_cpu])))
                else:
                    model.cache['train_prediction'] = np.vstack([y_cpu, y_hat_cpu])

        scheduler.step()
        epoch_lr = optimizer.param_groups[0]["lr"]
        val_loss, accuracy, accuracy1, accuracy2, MAE, MSE = validate_model(model, opt)
        end = time.time()
        print('epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {accuracy},'
              ' accuracy by 1 degree: {accuracy1}, accuracy by 2 degrees: {accuracy2}, MAE: {MAE},'
              ' MSE: {MSE}, time: {time}s, lr: {lr}'.format(
                epoch=epoch, train_loss=np.round(sum_loss / total, ROUND_CONST), val_loss=np.round(val_loss, ROUND_CONST),
                accuracy=np.round(accuracy, ROUND_CONST), accuracy1=np.round(accuracy1, ROUND_CONST),
                accuracy2=np.round(accuracy2, ROUND_CONST),MAE=np.round(MAE, ROUND_CONST),
                MSE=np.round(MSE, ROUND_CONST), time=int(end-start), lr=epoch_lr))
        if (epoch + 1) % 250 == 0:
            save_model(model, MAE)

    save_model(model, MAE)


def validate_model(model, opt):
    model.eval()
    total = 0
    sum_loss = 0.0
    pred = None
    actual = None
    for x, y in model.valid_loader:
        x, y, *data = model.unpack((x, y), device)
        y_cpu = y.cpu()
        actual = np.array(y_cpu) if actual is None else np.concatenate((actual, y_cpu))
        y_hat = model(x) if not data else model(x, data[0])

        loss = model.model_loss(y_hat, y, device)

        y_hat = model.predict(y_hat)
        y_hat_cpu = y_hat.cpu()
        pred = np.array(y_hat_cpu) if pred is None else np.concatenate((pred, y_hat_cpu))
        total += y.shape[0]
        scale_factor = (TEMP_SCALE * IR_TEMP_FACTOR) ** 2 if opt['normalize'] else 1
        sum_loss += loss.item() * y.shape[0] * scale_factor

    accuracy, accuracy1, accuracy2, MAE, MSE = metrics(pred, actual, opt)
    model.cache['accuracy'].append(accuracy)
    model.cache['MAE'].append(MAE)
    model.cache['MSE'].append(MSE)

    return sum_loss / total, accuracy, accuracy1, accuracy2, MAE, MSE


def get_best_model(model_name, criterion):
    if not model_name:
        return None, None

    listdir = os.listdir(MODELS_DIR)
    acceptable_models = re.compile('({model_name}.+mae[0-9\.]+\.pt)'.format(model_name=model_name))
    score_regex = re.compile('{model_name}.+mae([0-9\.]+)\.pt'.format(model_name=model_name))
    models = [re.search(acceptable_models, model).groups()[0] for model in listdir if re.findall(acceptable_models, model)]
    scores = [re.search(score_regex, model).groups()[0] for model in listdir if re.findall(score_regex, model)]

    if not models:
        return None, None

    idx = np.argmin(np.array(scores, dtype=float))
    inputs_dim = IRMaker.FRAME_WINDOW ** 2 * IRMaker.DATA_MAPS_COUNT + IRMaker.STATION_PARAMS_COUNT
    model = ModelFactory.create_model(model_name, None, None, None, inputs_dim, None, criterion).to(device)
    model.load_state_dict(torch.load('{dir}/{model}'.format(dir=MODELS_DIR, model=models[idx])))
    model.eval()
    # model = torch.load('{dir}/{model}'.format(dir=MODELS_DIR, model=models[idx]))
    # with open('{dir}/{model}'.format(dir=MODELS_DIR, model=models[idx]).replace(MODEL_EXTENSION, JSON_EXTENSION), 'r') as f:
    #     model.cache = json.loads(f.read())
    #     model.cache['train_prediction'] = np.array(model.cache['train_prediction'])
    return model, models[idx]


def present_distribution(opt):
    IRObjs = list()
    listdir = [dir for dir in os.listdir(BASE_DIR) if 'properties' not in dir and '.DS_Store' not in dir]
    for dir in listdir:
        IRObjs.append(IRMaker(dir, opt))
    border = np.max([np.abs(np.max(IRObj.IR)) for IRObj in IRObjs] + [np.abs(np.min(IRObj.IR)) for IRObj in IRObjs])
    bins = np.linspace(-border, border, 100)
    to_stack_histogram(IRObjs,
                       bins,
                       title='IR distribution histograms',
                       ylabel='Counts',
                       xlabel='IR value',
                       colors=['blue', 'red', 'orange', 'green', 'brown', 'pink', 'purple', 'cyan', 'brown', 'gray', 'olive']
                       )


def main(opt):
    # Create json station data for each folder
    csv_to_json('./resources/properties/data_table.csv')

    # Prepare data
    # X_train, y_train, X_valid, y_valid, means, loss_weights = prepare_data(opt)

    with open('test_data.json', 'r') as f:
        json_dict = json.loads(f.read())
    X_train, y_train, X_valid, y_valid, means, loss_weights = \
        np.array(json_dict['X_train']), np.array(json_dict['y_train']), np.array(json_dict['X_valid']),\
        np.array(json_dict['y_valid']), np.array(json_dict['means']), None

    batch_size = BATCH_SIZE if BATCH_SIZE else X_train.shape[0] // 10
    train_ds = Dataset(X_train, y_train)
    valid_ds = Dataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)

    if opt['model']:
        opt['model'].train_loader = train_dl
        opt['model'].valid_loader = valid_dl
        opt['model'].means = means
    else:
        model = ModelFactory.create_model(opt['model_name'], train_dl, valid_dl, means, X_train.shape[-1], loss_weights, opt['criterion']).to(device)
    model.cache['actual_mean'] = np.average(y_train) / IR_TEMP_FACTOR
    train_model(model, opt)
    return model


if __name__ == '__main__':
    # Choose Model: 'IRValue', 'IRClass', 'ConvNet', 'ResNet18', 'ResNet50', 'InceptionV3', 'VGG19', 'ResNetXt101'

    opt = {
        'to_train': True,
        'isCE': True,
        'criterion': nn.CrossEntropyLoss,
        'dirs': ['Zeelim_30.5.19_0630_E'],
        'model_name': 'ConvNet',
        'sampling_method': 'RFP',
        'samples': 5000,
        'exclude': False,
        'bias': None,
        'normalize': False,
        'label_kind': 'ir',
        'use_loss_weights': False
    }
    model, mae = get_best_model('', opt['criterion'])
    opt['model'] = model
    model = main(opt) if opt['to_train'] else model

    # create_graphs(model.cache)
    # present_distribution(opt)
    dirs = ['Zeelim_30.5.19_0630_E', 'Mishmar_3.3.20_1510_N']
    # IRMaker(dir).generate_image(model)
    # main('ConvNet', model, criterion, 'RFP', 10000, dirs, False, 'mean_ir')