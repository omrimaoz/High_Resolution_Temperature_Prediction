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
from sklearn.model_selection import train_test_split
from Dataset import Dataset
from Models import *


def train_model(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=model.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=model.lambda_scheduler)

    MAE = None

    for epoch in range(model.epochs):
        start = time.time()
        model.train()
        sum_loss = 0.0
        total = 0
        for batch, pack in enumerate(model.train_loader):
            x, y, *data = model.unpack(pack)
            y_pred = model(x, data[0]) if data else model(x)
            y_hat = model.predict(y_pred)

            loss = model.criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]

            if epoch == model.epochs -1:
                if model.cache['train_prediction'] is not None:
                    model.cache['train_prediction'] = np.hstack((model.cache['train_prediction'],
                                                                      np.vstack([y, y_hat])))
                else:
                    model.cache['train_prediction'] = np.vstack([y, y_hat])

        scheduler.step()
        epoch_lr = optimizer.param_groups[0]["lr"]
        val_loss, val_acc, MAE, MSE = validate_model(model)
        end = time.time()
        print('epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc},'
              ' MAE: {MAE}, MSE: {MSE}, time: {time}s, lr: {lr}'.format(
                epoch=epoch, train_loss=np.round(sum_loss / total, ROUND_CONST), val_loss=np.round(val_loss, ROUND_CONST),
                val_acc=np.round(val_acc, ROUND_CONST), MAE=np.round(MAE, ROUND_CONST),
                MSE=np.round(MSE, ROUND_CONST), time=int(end-start), lr=epoch_lr))
    path = '{dir}/{model}_{time}'.format(dir=MODELS_DIR, model=model.name,
                                         time=datetime.now().strftime('%d%m%y'))
    if MAE:
        path += '_mae{mae}'.format(mae=np.round(MAE, ROUND_CONST))
    torch.save(model, path + MODEL_EXTENSION)
    if model.cache['train_prediction'] is not None:
        model.cache['train_prediction'] = (model.cache['train_prediction'] / IR_TEMP_FACTOR).tolist()
        with open(path + JSON_EXTENSION, 'w') as f:
            f.write(json.dumps(model.cache))
        model.cache['train_prediction'] = np.array(model.cache['train_prediction'])


def validate_model(model):
    model.eval()
    total = 0
    sum_loss = 0.0
    pred = None
    actual = None
    for x, y in model.valid_loader:
        x, y, *data = model.unpack((x, y))
        actual = np.array(y) if actual is None else np.concatenate((actual, y))
        y_hat = model(x) if not data else model(x, data[0])

        loss = model.criterion(y_hat, y)

        y_hat = model.predict(y_hat)
        pred = np.array(y_hat) if pred is None else np.concatenate((pred, y_hat))
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]

    accuracy, MAE, MSE = metrics(pred, actual)
    model.cache['accuracy'].append(accuracy)
    model.cache['MAE'].append(MAE)
    model.cache['MSE'].append(MSE)

    return sum_loss / total, accuracy, MAE, MSE


def main(model_name, sample_method, dir_name=None):
    # Create json station data for each folder
    csv_to_json('./resources/properties/data_table.csv')

    # Prepare data
    X, y = prepare_data(10000, sample_method, dir_name)
    # X, y = prepare_data(50, 'SPP', dir_name)

    batch_size = BATCH_SIZE if BATCH_SIZE else X.shape[0] // 10
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    train_ds = Dataset(X_train, y_train)
    valid_ds = Dataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)

    model = ModelFactory.create_model(model_name, train_dl, valid_dl, X.shape[1], 6)
    model.cache['actual_mean'] = np.average(y_train) / IR_TEMP_FACTOR
    train_model(model)
    return model


if __name__ == '__main__':
    dir = 'Zeelim_30.5.19_0630_E'
    # model = get_best_model('ConvNet')
    model = None
    model = model if model else main('ConvNet', 'SFP')
    # create_graphs(model.cache)
    IRMaker(dir).generate_image(model)
