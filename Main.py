import time

from torch.utils.data import DataLoader
from torch import nn
import torch

from Prepare_Data import prepare_data
from utils import csv_to_json
from sklearn.model_selection import train_test_split
from Dataset import Dataset
from Model import Model

batch_size = 128
epochs = 50
lr = 0.001


def train_model(model, criterion, train_loader, valid_loader):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for epoch in range(epochs):
        start = time.time()
        model.train()
        sum_loss = 0.0
        total = 0
        for batch, pack in enumerate(train_loader):
            x, y = pack
            x = x.long()
            y = y.long()
            y_pred = model(x)

            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]

        val_loss, val_acc, precision, recall, F1 = validate_model(model, criterion, valid_loader)
        end = time.time()
        print('epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc},'
              ' precision: {precision}, recall: {recall}, F1: {F1}, time: {time}s'.format(
            epoch=epoch, train_loss=np.round(sum_loss/total, 3), val_loss=np.round(val_loss, 3),
            val_acc=np.round(val_acc, 3), precision=np.round(precision, 3), recall=np.round(recall, 3),
            F1=np.round(F1, 3), time=int(end-start)
        ))


def validate_model(model, criterion, valid_loader):
    model.eval()
    total = 0
    sum_loss = 0.0
    pred = None
    actual = None
    get_F1_score = get_binary_F1_score if model.num_classes == 2 else get_multi_F1_score
    for x, y, l in valid_loader:
        x = x.long()
        y = y.long()
        actual = np.array(y) if actual is None else np.concatenate((actual, y))
        y_hat = model(x, l)
        loss = criterion(y_hat, y)
        y_hat = y_hat.item()
        pred = np.array(y_hat) if pred is None else np.concatenate((pred, y_hat))
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
    accuracy, precision, recall, F1 = get_F1_score(actual, pred)
    return sum_loss / total, accuracy, precision, recall, F1


def main():
    # Create json station data for each folder
    csv_to_json('./resources/properties/data_table.csv')

    # Prepare data
    X, y = prepare_data(1000)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    train_ds = Dataset(X_train, y_train)
    valid_ds = Dataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)

    model = Model(X.shape[1])
    train_model(model, criterion=nn.CrossEntropyLoss(), train_loader=train_dl, valid_loader=val_dl)


if __name__ == '__main__':
    main()