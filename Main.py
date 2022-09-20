import argparse
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
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



def save_model(model, MAE, opt):
    model_name = model.name
    if opt['pretrained_ResNet18_correction']:
        model_name += '_Pretrained'
    path = '{dir}/{model}_{time}'.format(dir=MODELS_DIR, model=model_name,
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
    optimizer = torch.optim.Adam(parameters, lr=model.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=model.lambda_scheduler)
    epochs = opt["epochs"] if opt["epochs"] else model.epochs
    MAE = None

    for epoch in range(epochs):
        start = time.time()
        model.train()
        sum_loss = 0.0
        total = 0
        for batch, pack in enumerate(model.train_loader):
            x, y, *data = model.unpack(pack, device, opt)
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
        if (epoch + 1) % 5 == 0:
            save_model(model, MAE, opt)

    save_model(model, MAE, opt)


def validate_model(model, opt):
    model.eval()
    total = 0
    sum_loss = 0.0
    pred = None
    actual = None
    for x, y in model.valid_loader:
        x, y, *data = model.unpack((x, y), device, opt)
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


def get_best_model(model_name, opt):
    if not model_name:
        return None, None

    if opt['pretrained_ResNet18_correction']:
        model_name += '_Pretrained'

    listdir = os.listdir(MODELS_DIR)
    acceptable_models = re.compile('({model_name}_.*mae[0-9\.]+.*\.pt)'.format(model_name=model_name))
    score_regex = re.compile('{model_name}_.*mae([0-9\.]+).*\.pt'.format(model_name=model_name))
    models = [re.search(acceptable_models, model).groups()[0] for model in listdir if re.findall(acceptable_models, model)]
    scores = [re.search(score_regex, model).groups()[0] for model in listdir if re.findall(score_regex, model)]

    if not models:
        return None, None

    idx = np.argmin(np.array(scores, dtype=float))
    if opt['sampling_method'] in ['SPP', 'RPP']:
        inputs_dim = IRMaker.DATA_MAPS_COUNT + IRMaker.STATION_PARAMS_COUNT - opt['pretrained_ResNet18_correction']
    else:
        inputs_dim = IRMaker.FRAME_WINDOW ** 2 * (IRMaker.DATA_MAPS_COUNT - opt['pretrained_ResNet18_correction']) + IRMaker.STATION_PARAMS_COUNT
    model_name = model_name.replace('_Pretrained', '')
    model = ModelFactory.create_model(model_name, None, None, None, inputs_dim, None, opt['criterion'], opt).to(device)
    model.load_state_dict(torch.load('{dir}/{model}'.format(dir=MODELS_DIR, model=models[idx])))
    model.eval()
    # model = torch.load('{dir}/{model}'.format(dir=MODELS_DIR, model=models[idx]))
    # with open('{dir}/{model}'.format(dir=MODELS_DIR, model=models[idx]).replace(MODEL_EXTENSION, JSON_EXTENSION), 'r') as f:
    #     model.cache = json.loads(f.read())
    #     model.cache['train_prediction'] = np.array(model.cache['train_prediction'])
    return model, models[idx]


def present_distribution(opt):
    print('Plot Distribution')
    IRObjs = list()
    listdir = [dir for dir in os.listdir(BASE_DIR) if 'properties' not in dir and '.DS_Store' not in dir] if not opt['dirs'] else opt['dirs']
    for dir in listdir:
        IRObjs.append(IRMaker(dir, opt))
    border_top = int(np.max([np.max(IRObj.IR) for IRObj in IRObjs]))
    border_bottom = int(np.min([np.min(IRObj.IR) for IRObj in IRObjs]))
    bins = np.linspace(border_bottom, border_top, (border_top - border_bottom) * 5)
    to_stack_histogram(IRObjs,
                       bins,
                       title='IR distribution histograms',
                       ylabel='Counts',
                       xlabel='IR value',
                       colors=['blue', 'red', 'orange', 'green', 'brown', 'pink', 'purple', 'cyan', 'gray', 'olive']
                       )


def main(opt):
    print('Prepare Data')
    # Create json station data for each folder
    csv_to_json('./resources/properties/data_table.csv')

    # Prepare data
    X_train, y_train, X_valid, y_valid, means, loss_weights = prepare_data(opt)
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
        opt['model'] = ModelFactory.create_model(opt['model_name'], train_dl, valid_dl, means, X_train.shape[-1], loss_weights, opt['criterion'], opt).to(device)
    # opt['model'].cache['actual_mean'] = np.average(y_train) / IR_TEMP_FACTOR
    print('Start Training')
    train_model(opt['model'], opt)
    return model


def get_args():
    """A Helper function that defines the program arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--to_train', action='store_true')
    parser.add_argument('--dirs', type=str, default='["Mishmar_30.7.19_0640_E"]')
    parser.add_argument('--generate_dir', type=str, default='["Mishmar_30.7.19_0640_E"]')
    parser.add_argument('--model_name', type=str, default='ResNet18')
    parser.add_argument('--sampling_method', type=str, default='RFP')
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--use_pretrained_weights', action='store_true')
    parser.add_argument('--epochs', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Choose Model: 'IRValue', 'IRClass', 'ConvNet', 'ResNet18', 'ResNet50', 'InceptionV3', 'VGG19', 'ResNetXt101'
    opt = {
        'to_train': True,
        'isCE': True,
        'criterion': nn.CrossEntropyLoss,
        'dirs': ['Zeelim_30.5.19_0630_E'], #, 'Mishmar_30.7.19_0640_E', 'Mishmar_30.7.19_0820_S', 'Zeelim_23.9.19_1100_E',
                 # 'Mishmar_3.3.20_1510_N', 'Zeelim_7.11.19_1550_W', 'Zeelim_29.5.19_1730_W'],
        'generate_dir': ['Zeelim_30.5.19_0630_E'],
        'model_name': 'ConvNet',
        'sampling_method': 'RFP',
        'samples': 5000,
        'exclude': False,
        'bias': None,
        'normalize': False,
        'label_kind': 'ir',
        'use_loss_weights': False,
        'augmentation': False,
        'augmentation_p': 0.25,
        'augmentation_by_level': None, #np.array([6, 3, 0, 0, 0]),
        'use_pretrained_weights': False,
        'pretrained_ResNet18_correction': 0,
        "epochs": 10
    }
    args = get_args()
    for arg in vars(args):
        if arg == 'dirs' or arg == 'generate_dir':
            opt[arg] = json.loads(getattr(args, arg).replace("\'", '\"'))
        else:
            opt[arg] = getattr(args, arg)

    csv_to_json('./resources/properties/data_table.csv')
    opt['pretrained_ResNet18_correction'] = 3 if opt['use_pretrained_weights'] and opt['model_name'] == 'ResNet18' else 0
    model, model_name = get_best_model(opt['model_name'], opt)
    print('Found model: {}'.format(model_name))
    opt['model'] = model
    if opt['to_train']:
        main(opt)
        present_distribution(opt)

    # create_graphs(model.cache)

    IRObj = IRMaker(opt['generate_dir'][0], opt)
    # IRObj.generate_image(opt)
    IRObj.create_error_histogram(opt)
    IRObj.create_base_histogram(opt)
    IRObj.mark_weather_station_on_image(opt)
    IRObj.generate_error_images_discrete(opt, 5)
    IRObj.generate_error_images_continuous(opt)

