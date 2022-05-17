import time

import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from datetime import datetime
import torch

from IRMaker import IRMaker
from Prepare_Data import prepare_data
from utils import *
from sklearn.model_selection import train_test_split
from Dataset import Dataset
from Models import *


ROUND_CONST = 3
BATCH_SIZE = 0


def train_model(model, criterion, train_loader, valid_loader):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=model.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=model.lambda_scheduler)
    MAE = None
    for epoch in range(model.epochs):
        start = time.time()
        model.train()
        sum_loss = 0.0
        total = 0
        for batch, pack in enumerate(train_loader):
            x, y = model.unpack(pack)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]

        scheduler.step()
        epoch_lr = optimizer.param_groups[0]["lr"]
        val_loss, val_acc, MAE, MSE = validate_model(model, criterion, valid_loader)
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
    path += MODEL_EXTENSION
    torch.save(model, path)


def validate_model(model, criterion, valid_loader):
    model.eval()
    total = 0
    sum_loss = 0.0
    pred = None
    actual = None
    for x, y in valid_loader:
        x, y = model.unpack((x, y))
        actual = np.array(y) if actual is None else np.concatenate((actual, y))
        y_hat = model(x)

        loss = criterion(y_hat, y)

        y_hat = model.predict(y_hat)
        pred = np.array(y_hat) if pred is None else np.concatenate((pred, y_hat))
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]

    accuracy, MAE, MSE = metrics(pred, actual)
    return sum_loss / total, accuracy, MAE, MSE


def main():
    # Create json station data for each folder
    csv_to_json('./resources/properties/data_table.csv')

    # Prepare data
    X, y = prepare_data(50000)

    batch_size = BATCH_SIZE if BATCH_SIZE else X.shape[0] // 10
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    train_ds = Dataset(X_train, y_train)
    valid_ds = Dataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)

    ir_value_model = IRValue(X.shape[1])
    ir_class_model = IRClass(X.shape[1])

    # train_model(ir_value_model, criterion=nn.MSELoss(), train_loader=train_dl, valid_loader=val_dl)
    train_model(ir_class_model, criterion=nn.CrossEntropyLoss(), train_loader=train_dl, valid_loader=val_dl)

if __name__ == '__main__':
    dir = 'Zeelim_7.11.19_1550_W'
    main()
    model = get_best_model()
    IRMaker(dir).generate_image(model)
