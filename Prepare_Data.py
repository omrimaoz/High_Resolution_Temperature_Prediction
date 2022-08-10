import json
import os
import random

import tifffile as tiff
from PIL import Image
from collections import OrderedDict
import numpy as np
from torch import nn

from IRMaker import IRMaker
from utils import *
from Augmentation import *

data_map = OrderedDict({
    'Height': None,
    'RealSolar': None,
    'Shade': None,
    'SkyView': None,
    'SLP': None,
    'TGI': None,
})


def augmentation(X, y, opt):
    filters = [rotate_90, rotate_270, vflip, noise] # [rotate_90, rotate_180, rotate_270, vflip, hflip, noise]
    aug_X = np.zeros(shape=(X.shape[0] * len(filters), X.shape[1]))
    aug_y = np.zeros(shape=(X.shape[0] * len(filters)))

    for i in range(X.shape[0]):
        frame = X[i].reshape(IRMaker.DATA_MAPS_COUNT, IRMaker.FRAME_WINDOW, IRMaker.FRAME_WINDOW)
        for j, filter in enumerate(filters):
            aug_X[i * len(filters) + j] = filter(frame, opt['augmentation_p']).flatten()
            aug_y[i * len(filters) + j] = y[i]

    return aug_X, aug_y


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
        if image.dtype == np.int64:
            scale_factor = 1
        else:
            scale_factor = TEMP_SCALE * IR_TEMP_FACTOR
        num_levels = int((high - low) / (IR_TEMP_DIFF / scale_factor))
        num_level_sample = num_samples // num_levels

        train_indices = list()
        valid_indices = list()
        levels = [list() for _ in range(num_levels + 1)]
        for i in range(image.shape[0]):
            for j in range(image.shape[0]):
                levels[int((image[i][j] - low) // (IR_TEMP_DIFF / scale_factor))].append((i, j))

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


def pixel_to_pixel_sampling(opt, listdir, method):
    input_image_num = IRMaker.DATA_MAPS_COUNT if opt['label_kind'] == 'ir' else 3
    dtype = np.int if opt['isCE'] else np.float
    X_train = np.zeros(shape=(int(opt['samples'] * len(listdir)), input_image_num), dtype=np.float)
    y_train = np.zeros(shape=(int(opt['samples'] * len(listdir))), dtype=dtype)
    X_valid = np.zeros(shape=(int(opt['samples'] * len(listdir)), input_image_num), dtype=np.float)
    y_valid = np.zeros(shape=(int(opt['samples'] * len(listdir))), dtype=dtype)
    m, n = 0, 0
    means = list()
    loss_weights = np.zeros(TEMP_SCALE * IR_TEMP_FACTOR)

    for dir in listdir:
        IRObj = IRMaker(dir, opt)
        loss_weights += IRObj.loss_weights
        dir_data = IRObj.get_data_dict()
        station_data = IRObj.station_data
        label_data = IRObj.IR
        means.append(np.average(IRObj.IR))

        train_row, train_col, valid_row, valid_col = random_sampling_by_method(method, IRObj.IR, opt['samples'])

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

    loss_weights = loss_weights / len(listdir)

    return X_train[:m], y_train[:m], X_valid[:n], y_valid[:n], means, loss_weights


def frame_to_pixel_sampling(opt, listdir, method):
    input_image_num = IRMaker.DATA_MAPS_COUNT if opt['label_kind'] == 'ir' else 3
    dtype = np.int
    X_train = np.zeros(shape=(int(opt['samples'] * len(listdir)), 2), dtype=dtype)
    y_train = np.zeros(shape=(int(opt['samples'] * len(listdir))), dtype=dtype)
    X_valid = np.zeros(shape=(int(opt['samples'] * len(listdir)), 2), dtype=dtype)
    y_valid = np.zeros(shape=(int(opt['samples'] * len(listdir))), dtype=dtype)
    m, n = 0, 0
    means = list()
    loss_weights = np.zeros(TEMP_SCALE * IR_TEMP_FACTOR) if opt['use_loss_weights']  else None

    for dir in listdir:
        IRObj = IRMaker(dir, opt)
        loss_weights = loss_weights + IRObj.loss_weights if opt['use_loss_weights'] else None
        # dir_data = [np.pad(image, IRMaker.FRAME_RADIUS) for image in IRObj.get_data_dict()] if opt['label_kind'] == 'ir' else \
        #     [np.pad(IRObj.RGB, pad_width=((IRMaker.FRAME_RADIUS, IRMaker.FRAME_RADIUS), (IRMaker.FRAME_RADIUS, IRMaker.FRAME_RADIUS), (0,0)))]
        # station_data = IRObj.station_data
        # mean_ir = 0
        # means.append(np.average(IRObj.IR))

        train_row, train_col, valid_row, valid_col = random_sampling_by_method(method, IRObj.IR, opt['samples'])

        for i, j in zip(train_row, train_col):
            # data_samples = list()
            # for image in dir_data:
            #     frame = IRMaker.get_frame(image, i, j).flatten()
            #     data_samples.extend(frame)
            # if opt['label_kind'] == 'mean_ir':
            #     mean_ir = np.average(IRMaker.get_frame(IRObj.IR, i, j))
            # for key in IRObj.STATION_PARAMS_TO_USE:
            #     data_samples.append(station_data[key])
            X_train[m] = np.array([i,j])
            y_train[m] = IRObj.IR[i][j] if opt['label_kind'] == 'ir' else mean_ir
            m += 1

        for i, j in zip(valid_row, valid_col):
            # data_samples = list()
            # for k, image in enumerate(dir_data):
            #     frame = IRMaker.get_frame(image, i, j).flatten()
            #     data_samples.extend(frame)
            # if opt['label_kind'] == 'mean_ir':
            #     mean_ir = np.average(IRMaker.get_frame(IRObj.IR, i, j))
            # for key in IRObj.STATION_PARAMS_TO_USE:
            #     data_samples.append(station_data[key])
            X_valid[n] = np.array([i,j])
            y_valid[n] = IRObj.IR[i][j] if opt['label_kind'] == 'ir' else mean_ir
            n += 1

    loss_weights = loss_weights / len(listdir) if opt['use_loss_weights'] else None

    X_train, y_train = X_train[:m], y_train[:m]
    if opt['augmentation']:
        aug_X_train, aug_y_train = augmentation(X_train, y_train, opt)
        X_train, y_train = np.vstack((X_train, aug_X_train)), np.hstack((y_train, aug_y_train))

    return X_train, y_train, X_valid[:n], y_valid[:n], means, loss_weights


def prepare_data(opt):
    listdir = [dir for dir in os.listdir(BASE_DIR) if 'properties' not in dir and '.DS_Store' not in dir]
    if opt['dirs']:
        if opt['exclude']:
            [listdir.remove(dir) for dir in opt['dirs']]
        else:
            listdir = opt['dirs']

    for dir in listdir:
        if not os.path.exists('{base_dir}/{dir}/station_data.json'.format(base_dir=BASE_DIR, dir=dir)):
            listdir.remove(dir)

    if opt['model_name'] == 'InceptionV3':
        IRMaker.FRAME_RADIUS, IRMaker.FRAME_WINDOW = 149, 299
    if opt['model_name'] == 'VGG19':
        IRMaker.FRAME_RADIUS, IRMaker.FRAME_WINDOW = 112, 224
    '''
    inputs to consider:
        - From data: Height (DSM), ..
        - From Station: Julian Day, Day Time, Habitat, Wind Speed, Air Temperature, Ground Temperature Humidity, Pressure, Radiation.   
    '''

    inputs = IRMaker.DATA_MAPS_COUNT + IRMaker.STATION_PARAMS_COUNT
    if opt['sampling_method'] == 'SPP':
        X_train, y_train, X_valid, y_valid, means, loss_weights = pixel_to_pixel_sampling(opt, listdir, 'Simple')
    if opt['sampling_method'] == 'RPP':
        X_train, y_train, X_valid, y_valid, means, loss_weights = pixel_to_pixel_sampling(opt, listdir, 'Relative')
    if opt['sampling_method'] == 'SFP':
        X_train, y_train, X_valid, y_valid, means, loss_weights = frame_to_pixel_sampling(opt, listdir, 'Simple')
    if opt['sampling_method'] == 'RFP':
        inputs = IRMaker.DATA_MAPS
        X_train, y_train, X_valid, y_valid, means, loss_weights = frame_to_pixel_sampling(opt, listdir, 'Relative')

    return X_train, y_train, X_valid, y_valid, means, loss_weights
