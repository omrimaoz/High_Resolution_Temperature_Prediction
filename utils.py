import csv
import json
import os
import re

import numpy as np
import tifffile as tiff
import torch

ROUND_CONST = 3
DEGREE_ERROR = 2

BASE_DIR = './resources'
MODELS_DIR = './models'
CHECKPOINTS_DIR = './checkpoints'
MODEL_EXTENSION = '.pt'


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
        for i in range(1, len(row)):
            if row[i] in locations:
                data_json[fields[i]] = locations[row[i]]
            else:
                data_json[fields[i]] = float(row[i]) / 10 if row[i] else 0.
        for option in ["", "_W", "_E", "_N", "_S"]:
            if row[0] + option in os.listdir("./resources"):
                with open("./resources/{dir}/station_data.json".format(dir=row[0] + option), 'w') as data_file:
                    json.dump(data_json, data_file, indent=4, separators=(',', ': '))


def metrics(predictions, actuals):
    MAE = float(np.average(np.abs(actuals - predictions)))  # Mean Absolute Error - Average Euclidean distances between two points
    MSE = float(np.average(np.power(actuals - predictions, 2)))
    accuracy = float(np.sum(((np.abs(actuals - predictions) < DEGREE_ERROR) * 1.)) / actuals.shape[0])
    return accuracy, MAE, MSE


def evaluate_prediceted_IR(dir):
    RealIR = tiff.imread('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=dir))
    PredictedIR = tiff.imread('{base_dir}/{dir}/PredictedIR.tif'.format(base_dir=BASE_DIR, dir=dir))

    Accuracy, MAE, MSE = metrics(PredictedIR, RealIR)
    print('IRMaker Result: Accuracy: {Accuracy}, MAE: {MAE}, MSE: {MSE}'.format(
        Accuracy=np.round(Accuracy, ROUND_CONST), MAE=np.round(MAE, ROUND_CONST),
        MSE=np.round(MSE, ROUND_CONST)
    ))

def get_best_model():
    listdir = os.listdir(MODELS_DIR)
    acceptable_models = re.compile('(.+mae[0-9\.]+\.pt)')
    score_regex = re.compile('mae([0-9\.]+)\.pt')
    models = [re.search(acceptable_models, model).groups()[0] for model in listdir if re.findall(acceptable_models, model)]
    scores = [re.search(score_regex, model).groups()[0] for model in listdir if re.findall(score_regex, model)]
    idx = np.argmin(np.array(scores, dtype=float))
    model = torch.load('{dir}/{model}'.format(dir=MODELS_DIR, model=models[idx]))
    return model




