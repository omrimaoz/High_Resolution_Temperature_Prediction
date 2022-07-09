import json
import os
import random

import tifffile as tiff
from PIL import Image
from collections import OrderedDict
import numpy as np
from IRMaker import IRMaker
from utils import *

data_map = OrderedDict({
    'Height': None,
    'RealSolar': None,
    'Shade': None,
    'SkyView': None,
    'SLP': None,
    'TGI': None,
})


def get_frame(image, i, j):
    return image[i: i + IRMaker.FRAME_WINDOW, j: j + IRMaker.FRAME_WINDOW]


def random_sampling_by_method(method, image, num_samples):
    if method == 'Simple':
        row_indices = np.random.choice(image.shape[0], num_samples)
        col_indices = np.random.choice(image.shape[0], num_samples)
        indices = list(zip(row_indices, col_indices))
        random.shuffle(indices)
        train_row, train_col = zip(*indices[:int(len(indices) * (1 - SPLIT_FACTOR))])
        valid_row, valid_col = zip(*indices[int(len(indices) * (1 - SPLIT_FACTOR)):])
        return train_row, train_col, valid_row, valid_col

    if method == 'Relative':
        low = np.min(image)
        high = np.max(image)
        num_levels = int((high - low) / (IR_TEMP_DIFF / (TEMP_SCALE * IR_TEMP_FACTOR)))
        num_level_sample = num_samples // num_levels

        train_indices = list()
        valid_indices = list()
        levels = [list() for _ in range(num_levels + 1)]
        for i in range(image.shape[0]):
            for j in range(image.shape[0]):
                levels[int((image[i][j] - low) // (IR_TEMP_DIFF / (TEMP_SCALE * IR_TEMP_FACTOR)))].append((i, j))

        forward_sample = 0
        for i in range(len(levels)):
            current_sample = num_level_sample + forward_sample
            if current_sample > len(levels[i]):
                forward_sample = current_sample - len(levels[i])
            else:
                forward_sample = 0
            indices = random.sample(levels[i], current_sample - forward_sample)
            random.shuffle(indices)
            train_indices += indices[:int(np.round(len(indices) * (1 - SPLIT_FACTOR)))]
            valid_indices += indices[int(np.round(len(indices) * (1 - SPLIT_FACTOR))):]

        train_row, train_col = zip(*train_indices)
        valid_row, valid_col = zip(*valid_indices)
        return train_row, train_col, valid_row, valid_col


def pixel_to_pixel_sampling(num_samples, inputs, listdir, method):
    X_train = np.zeros(shape=(int(num_samples * len(listdir)), inputs), dtype=np.float)
    y_train = np.zeros(shape=(int(num_samples * len(listdir))), dtype=np.float)
    X_valid = np.zeros(shape=(int(num_samples * len(listdir)), inputs), dtype=np.float)
    y_valid = np.zeros(shape=(int(num_samples * len(listdir))), dtype=np.float)
    m, n = 0, 0
    means = list()

    for dir in listdir:
        IRObj = IRMaker(dir, train=True)
        dir_data = IRObj.get_data_dict()
        station_data = IRObj.station_data
        label_data = IRObj.IR
        means.append(np.average(IRObj.IR))

        train_row, train_col, valid_row, valid_col = random_sampling_by_method(method, IRObj.IR, num_samples)

        for i, j in zip(train_row, train_col):
            data_samples = list()
            for image in dir_data:
                data_samples.append(image[i][j])
            for key in IRObj.STATION_PARAMS_TO_USE:
                data_samples.append(station_data[key])
            X_train[m] = np.array(data_samples)
            y_train[m] = label_data[i][j]
            m += 1

        for i, j in zip(valid_row, valid_col):
            data_samples = list()
            for image in dir_data:
                data_samples.append(image[i][j])
            for key in IRObj.STATION_PARAMS_TO_USE:
                data_samples.append(station_data[key])
            X_valid[n] = np.array(data_samples)
            y_valid[n] = label_data[i][j]
            n += 1

    return X_train[:m], y_train[:m], X_valid[:n], y_valid[:n], means


def frame_to_pixel_sampling(num_samples, inputs, listdir, method):
    X_train = np.zeros(shape=(int(num_samples * len(listdir)), IRMaker.DATA_MAPS_COUNT * (IRMaker.FRAME_WINDOW ** 2) + IRMaker.STATION_PARAMS_COUNT), dtype=np.float)
    y_train = np.zeros(shape=(int(num_samples * len(listdir))), dtype=np.float)
    X_valid = np.zeros(shape=(int(num_samples * len(listdir)), IRMaker.DATA_MAPS_COUNT * (IRMaker.FRAME_WINDOW ** 2) + IRMaker.STATION_PARAMS_COUNT), dtype=np.float)
    y_valid = np.zeros(shape=(int(num_samples * len(listdir))), dtype=np.float)
    m, n = 0, 0
    means = list()

    for dir in listdir:
        IRObj = IRMaker(dir, train=True)
        dir_data = [np.pad(image, IRMaker.FRAME_RADIUS) for image in IRObj.get_data_dict()]
        station_data = IRObj.station_data
        label_data = IRObj.IR
        means.append(np.average(IRObj.IR))

        train_row, train_col, valid_row, valid_col = random_sampling_by_method(method, IRObj.IR, num_samples)

        for i, j in zip(train_row, train_col):
            data_samples = list()
            for image in dir_data:
                frame = get_frame(image, i, j).flatten()
                data_samples.extend(frame)
            for key in IRObj.STATION_PARAMS_TO_USE:
                data_samples.append(station_data[key])
            X_train[m] = np.array(data_samples)
            y_train[m] = label_data[i][j]
            m += 1

        for i, j in zip(valid_row, valid_col):
            data_samples = list()
            for k, image in enumerate(dir_data):
                frame = get_frame(image, i, j).flatten()
                data_samples.extend(frame)
            for key in IRObj.STATION_PARAMS_TO_USE:
                data_samples.append(station_data[key])
            X_valid[n] = np.array(data_samples)
            y_valid[n] = label_data[i][j]
            n += 1

    return X_train[:m], y_train[:m], X_valid[:n], y_valid[:n], means


def prepare_data(model_name, num_samples, sampling_method, dir, exclude):
    listdir = [dir for dir in os.listdir(BASE_DIR) if 'properties' not in dir and '.DS_Store' not in dir]
    if dir:
        if exclude:
            listdir.remove(dir)
        else:
            listdir = [dir]

    for dir in listdir:
        if not os.path.exists('{base_dir}/{dir}/station_data.json'.format(base_dir=BASE_DIR, dir=dir)):
            listdir.remove(dir)

    if model_name == 'InceptionV3':
        IRMaker.FRAME_RADIUS, IRMaker.FRAME_WINDOW = 149, 299
    if model_name == 'VGG19':
        IRMaker.FRAME_RADIUS, IRMaker.FRAME_WINDOW = 112, 224
    '''
    inputs to consider:
        - From data: Height (DSM), ..
        - From Station: Julian Day, Day Time, Habitat, Wind Speed, Air Temperature, Ground Temperature Humidity, Pressure, Radiation.   
    '''

    inputs = IRMaker.DATA_MAPS_COUNT + IRMaker.STATION_PARAMS_COUNT
    if sampling_method == 'SPP':
        X_train, y_train, X_valid, y_valid, means = pixel_to_pixel_sampling(num_samples, inputs, listdir, 'Simple')
    if sampling_method == 'RPP':
        X_train, y_train, X_valid, y_valid, means = pixel_to_pixel_sampling(num_samples, inputs, listdir, 'Relative')
    if sampling_method == 'SFP':
        X_train, y_train, X_valid, y_valid, means = frame_to_pixel_sampling(num_samples, inputs, listdir, 'Simple')
    if sampling_method == 'RFP':
        inputs = IRMaker.DATA_MAPS
        X_train, y_train, X_valid, y_valid, means = frame_to_pixel_sampling(num_samples, inputs, listdir, 'Relative')

    return X_train, y_train, X_valid, y_valid, means
