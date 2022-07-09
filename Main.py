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


def train_model(model):
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

            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]

            if epoch == model.epochs -1:
                y_cpu, y_hat_cpu = y.cpu(), y_hat.cpu()
                if model.cache['train_prediction'] is not None:
                    model.cache['train_prediction'] = np.hstack((model.cache['train_prediction'],
                                                                      np.vstack([y_cpu, y_hat_cpu])))
                else:
                    model.cache['train_prediction'] = np.vstack([y_cpu, y_hat_cpu])

        scheduler.step()
        epoch_lr = optimizer.param_groups[0]["lr"]
        val_loss, val_acc, MAE, MSE = validate_model(model)
        end = time.time()
        print('epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc},'
              ' MAE: {MAE}, MSE: {MSE}, time: {time}s, lr: {lr}'.format(
                epoch=epoch, train_loss=np.round(sum_loss / total, ROUND_CONST), val_loss=np.round(val_loss, ROUND_CONST),
                val_acc=np.round(val_acc, ROUND_CONST), MAE=np.round(MAE, ROUND_CONST),
                MSE=np.round(MSE, ROUND_CONST), time=int(end-start), lr=epoch_lr))
        if (epoch + 1) % 1000 == 0:
            save_model(model, MAE)

    save_model(model, MAE)


def validate_model(model):
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
        sum_loss += loss.item() * y.shape[0] * 210 ** 2

    accuracy, MAE, MSE = metrics(pred, actual)
    model.cache['accuracy'].append(accuracy)
    model.cache['MAE'].append(MAE)
    model.cache['MSE'].append(MSE)

    return sum_loss / total, accuracy, MAE, MSE

def get_best_model(model_name):
    if not model_name:
        return None

    listdir = os.listdir(MODELS_DIR)
    acceptable_models = re.compile('({model_name}.+mae[0-9\.]+\.pt)'.format(model_name=model_name))
    score_regex = re.compile('{model_name}.+mae([0-9\.]+)\.pt'.format(model_name=model_name))
    models = [re.search(acceptable_models, model).groups()[0] for model in listdir if re.findall(acceptable_models, model)]
    scores = [re.search(score_regex, model).groups()[0] for model in listdir if re.findall(score_regex, model)]

    if not models:
        return None

    idx = np.argmin(np.array(scores, dtype=float))
    inputs_dim = IRMaker.FRAME_WINDOW ** 2 * IRMaker.DATA_MAPS_COUNT + IRMaker.STATION_PARAMS_COUNT
    model = ModelFactory.create_model(model_name, None, None, None, inputs_dim, 1, criterion).to(device)
    model.load_state_dict(torch.load('{dir}/{model}'.format(dir=MODELS_DIR, model=models[idx])))
    model.eval()
    # model = torch.load('{dir}/{model}'.format(dir=MODELS_DIR, model=models[idx]))
    with open('{dir}/{model}'.format(dir=MODELS_DIR, model=models[idx]).replace(MODEL_EXTENSION, JSON_EXTENSION), 'r') as f:
        model.cache = json.loads(f.read())
        model.cache['train_prediction'] = np.array(model.cache['train_prediction'])
    return model


def main(model_name, model, criterion, sampling_method, samples=5000, dir_name=None, exclude=False):
    # Create json station data for each folder
    csv_to_json('./resources/properties/data_table.csv')

    # Prepare data
    X_train, y_train, X_valid, y_valid, means = prepare_data(model_name, samples, sampling_method, dir_name, exclude)

    batch_size = BATCH_SIZE if BATCH_SIZE else X_train.shape[0] // 10
    train_ds = Dataset(X_train, y_train)
    valid_ds = Dataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)

    if model:
        model.train_loader = train_dl
        model.valid_loader = valid_dl
        model.means = means
        model.criterion = criterion
    else:
        model = ModelFactory.create_model(model_name, train_dl, valid_dl, means, X_train.shape[-1], 1, criterion).to(device)
    model.cache['actual_mean'] = np.average(y_train) / IR_TEMP_FACTOR
    train_model(model)
    return model


if __name__ == '__main__':
    # Choose Model: 'IRValue', 'IRClass', 'ConvNet', 'ResNet18', 'ResNet50', 'InceptionV3', 'VGG19', 'ResNetXt101'
    to_train = True
    criterion = WMSELoss
    dir = 'Zeelim_7.11.19_1550_W'
    model = get_best_model('ConvNet')
    model = main('ConvNet', model, criterion, 'RFP', 100000, dir, False) if to_train else model
    # create_graphs(model.cache)
    dir = 'Zeelim_7.11.19_1550_W'
    # IRMaker(dir).generate_image(model)
