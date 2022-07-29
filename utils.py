import csv
import json
import os
import re

import numpy as np
import tifffile as tiff
import torch
from matplotlib import pyplot as plt
from torch import nn

ROUND_CONST = 3
BATCH_SIZE = 50
DEGREE_ERROR = 0.5
IR_TEMP_FACTOR = 10
IR_TEMP_DIFF = 0.2
SPLIT_FACTOR = 0.2
SIMILARITY_THRESHOLD = 0.3
TEMP_SCALE = 70
POSITIVE_CONST = 70

TO_GRAPH = True

BASE_DIR = './resources'
MODELS_DIR = './models'
CHECKPOINTS_DIR = './checkpoints'
MODEL_EXTENSION = '.pt'
JSON_EXTENSION = '.json'


def csv_to_json(path):
    rows = []
    locations = {
        'desert': 0.,
        'mediterranean': 1.
    }
    # read csv file
    with open(path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        fields = next(csv_reader)
        for row in csv_reader:
            rows.append(row)

    # generate json file for each row in the appropriate directory
    for row in rows:
        data_json = {}
        time = row[0].split('_')[-1]
        time = int(time[:2]) * 6 + int(time[2:]) // 10
        data_json['time'] = float(time)
        for i in range(2, len(row)):
            if row[i] in locations:
                data_json[fields[i]] = locations[row[i]]
            else:
                if 'temp' in fields[i]:
                    data_json[fields[i]] = (float(row[i]) - 273.5) / (TEMP_SCALE * IR_TEMP_FACTOR)
                elif not row[i] or float(row[i]) < 500:
                    data_json[fields[i]] = float(row[i]) / 500 if row[i] else 0.
                else:
                    data_json[fields[i]] = float(row[i]) / 2000 if row[i] else 0.
        for option in ["", "_W", "_E", "_N", "_S"]:
            if row[0] + option in os.listdir("./resources"):
                with open("./resources/{dir}/station_data.json".format(dir=row[0] + option), 'w') as data_file:
                    json.dump(data_json, data_file, indent=4, separators=(',', ': '))


def metrics(predictions, actuals, opt):
    upscale_factor = TEMP_SCALE * IR_TEMP_FACTOR if opt['normalize'] and opt['criterion'] != nn.CrossEntropyLoss else 1
    downscale_factor = 1 / IR_TEMP_FACTOR if opt['isCE'] else upscale_factor
    MAE = float(np.average(np.abs(actuals - predictions))) * upscale_factor
    MSE = float(np.average(np.power(actuals - predictions, 2))) * upscale_factor ** 2
    accuracy = float(np.sum(((np.abs(actuals - predictions) < DEGREE_ERROR / downscale_factor) * 1.)) / actuals.shape[0])
    accuracy1 = float(np.sum(((np.abs(actuals - predictions) < 1 / downscale_factor) * 1.)) / actuals.shape[0])
    accuracy2 = float(np.sum(((np.abs(actuals - predictions) < 2 / downscale_factor) * 1.)) / actuals.shape[0])

    return accuracy, accuracy1, accuracy2, MAE, MSE


def evaluate_prediceted_IR(dir):
    RealIR = tiff.imread('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=dir))
    PredictedIR = tiff.imread('{base_dir}/{dir}/PredictedIR.tif'.format(base_dir=BASE_DIR, dir=dir))

    Accuracy, MAE, MSE = metrics(PredictedIR.flatten() * IR_TEMP_FACTOR, RealIR.flatten() * IR_TEMP_FACTOR)
    print('IRMaker Result: Accuracy: {Accuracy}, MAE: {MAE}, MSE: {MSE}'.format(
        Accuracy=np.round(Accuracy, ROUND_CONST), MAE=np.round(MAE, ROUND_CONST),
        MSE=np.round(MSE, ROUND_CONST)
    ))


def to_graph(y, x, title, ylabel, xlabel, colors, markers, labels, v_val=None, v_label=None):
    for i, t in enumerate(x):
        plt.scatter(x[i], y, c=colors[i], marker=markers[i], label=labels[i])
    if v_val:
        plt.axvline(x=v_val, color='orange', linestyle='--', lw=4, label='{}={}'.format(v_label, v_val))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc=0)
    plt.show()  # uncomment if you want to show graphs


def to_histogram(x, bins, title, ylabel, xlabel, color, v_val=None, v_label=None):
    plt.hist(x, bins, color=color)
    if v_val:
        plt.axvline(x=v_val, color='orange', linestyle='--', lw=4, label='{}={}'.format(v_label, v_val))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc=0)
    plt.show()  # uncomment if you want to show graphs


def to_stack_histogram(IRObjs, bins, title, ylabel, xlabel, colors, v_val=None, v_label=None):
    for i in range(len(IRObjs)):
        plt.hist(IRObjs[i].IR.flatten(), bins, alpha=0.5, label=IRObjs[i].dir, color=colors[i])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc='upper right')
    plt.show()


def create_graphs(cache):
    '''
    1st graph: Accuracy as function of MAE/MSE
    '''
    to_graph(y=cache['accuracy'],
             x=[cache['MAE'], cache['MSE']],
             title='Accuracy as function of MAE/MSE',
             ylabel='Accuracy',
             xlabel='MAE/MSE',
             colors=['b', 'r'],
             markers=['.', '*'],
             labels=['MAE', 'MSE']
             )

    '''
    2nd graph: Function of diff (prediction-actual) as actual temperature
    '''
    train_validation = cache['train_prediction'].copy()
    train_validation[1] = train_validation[1] - train_validation[0]
    train_validation = train_validation[:, train_validation[0].argsort()]
    to_graph(y=train_validation[1],
             x=[train_validation[0]],
             title='Function of diff (prediction-actual) as actual temperature',
             ylabel='Prediction - Actual',
             xlabel='Actual temperature value',
             colors=['b'],
             markers=['.'],
             labels=['diff (prediction-actual)'],
             v_val=cache['actual_mean'],
             v_label='Actual Mean'
             )

    '''
    3nd graph: diff |prediction-actual| histogram
    '''
    train_validation = cache['train_prediction'].copy()
    train_validation[1] = np.abs(train_validation[1] - train_validation[0])
    train_validation = train_validation[:, train_validation[0].argsort()]
    to_histogram(
        x=train_validation[1],
        bins=int((np.max(train_validation[1]) - np.min(train_validation[1]))/IR_TEMP_DIFF),
        title='diff |prediction-actual| histogram',
        ylabel='counts',
        xlabel='diff |prediction-actual|',
        color='b',
    )

    '''
    4nd graph: Error as function of actual temperature values histogram
    '''
    train_validation = cache['train_prediction'].copy()
    train_validation[1] = np.abs(train_validation[1] - train_validation[0])
    train_validation = train_validation[:, train_validation[0].argsort()]
    train_validation = train_validation[:, train_validation[1] > 2]

    to_histogram(
        x=train_validation[0],
        bins=int((np.max(train_validation[0]) - np.min(train_validation[0]))/IR_TEMP_DIFF),
        title='Error as function of actual temperature values histogram',
        ylabel='counts of errors',
        xlabel='actual temperature values ',
        color='b',
        v_val=cache['actual_mean'],
        v_label='Actual Mean'
    )

    '''
    5nd graph: Pixels error (MSE) as function of distance from mean (require the data to consider a single register image)
    '''
    train_validation = cache['train_prediction'].copy()
    train_validation[0] = (train_validation[0] - train_validation[1]) ** 2
    train_validation[1] = np.abs(cache['actual_mean'] - train_validation[1])
    # train_validation[1] = train_validation[1] -
    train_validation = train_validation[:, train_validation[0].argsort()]

    to_graph(y=train_validation[0],
             x=[train_validation[1]],
             title='Pixels error (MSE) as function of distance from mean',
             ylabel='Pixels error (MSE)',
             xlabel='Pixels distance from mean',
             colors=['b'],
             markers=['.'],
             labels=['Pixels error (MSE)']
             )
