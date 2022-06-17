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

def random_sampling_by_method(method, image, num_samples):
    if method == 'Simple':
        row_indices = np.random.choice(image.shape[0], num_samples)
        col_indices = np.random.choice(image.shape[0], num_samples)
        return row_indices, col_indices

    if method == 'Relative':
        low = np.min(image)
        high = np.max(image)
        num_level_sample = int(num_samples / ((high - low) / IR_TEMP_DIFF))

        indices = list()
        levels = [list() for _ in range(int((high - low) // IR_TEMP_DIFF) + 1)]
        for i in range(image.shape[0]):
            for j in range(image.shape[0]):
                levels[int((image[i][j] - low) // IR_TEMP_DIFF)].append((i, j))

        forward_sample = 0
        for i in range(len(levels)):
            current_sample = num_level_sample + forward_sample
            if current_sample > len(levels[i]):
                forward_sample = current_sample - len(levels[i])
            else:
                forward_sample = 0
            indices += random.sample(levels[i], current_sample - forward_sample)

        return zip(*indices)


def pixel_to_pixel_sampling(num_samples, inputs, listdir, method):
    X = np.zeros(shape=(num_samples * len(listdir), inputs), dtype=np.float)
    y = np.zeros(shape=(num_samples * len(listdir)), dtype=np.float)
    k = 0

    for dir in listdir:
        IRObj = IRMaker(dir, train=True)
        dir_data = IRObj.get_data_dict()
        station_data = IRObj.station_data
        label_data = IRObj.IR

        row_indices, col_indices = random_sampling_by_method(method, IRObj.IR, num_samples)

        for i, j in zip(row_indices, col_indices):
            data_samples = list()
            for image in dir_data:
                data_samples.append(image[i][j])
            for key in IRObj.STATION_PARAMS_TO_USE:
                data_samples.append(station_data[key])
            X[k] = np.array(data_samples)
            y[k] = label_data[i][j]
            k += 1
            if dir_data[0].shape[0] != 1000:
                print(1)

    return X[:k], y[:k]


def frame_to_pixel_sampling(num_samples, inputs, listdir, method):
    X = np.zeros(shape=(num_samples * len(listdir), inputs + ((2 * FRAME_RADIUS + 1) ** 2 - 1) * 6), dtype=np.float)  # TODO replace 6 with param
    y = np.zeros(shape=(num_samples * len(listdir)), dtype=np.float)
    k = 0

    for dir in listdir:
        IRObj = IRMaker(dir, train=True)
        dir_data = [np.pad(image, FRAME_RADIUS) for image in IRObj.get_data_dict()]
        station_data = IRObj.station_data
        label_data = IRObj.IR

        row_indices, col_indices = random_sampling_by_method(method, IRObj.IR, num_samples)

        for i, j in zip(row_indices, col_indices):
            data_samples = list()
            for image in dir_data:
                flat_image = get_frame(image, i + FRAME_RADIUS, j + FRAME_RADIUS, FRAME_RADIUS).flatten()
                data_samples.extend(flat_image)
            for key in IRObj.STATION_PARAMS_TO_USE:
                data_samples.append(station_data[key])
            X[k] = np.array(data_samples)
            y[k] = label_data[i][j]
            k += 1
            if dir_data[0].shape[0] != 1000 and dir_data[0].shape[0] != 1000 + FRAME_RADIUS*2:
                print(1)

    return X, y


def prepare_data(num_samples, sampling_method, dir):
    listdir = [dir] if dir else [dir for dir in os.listdir(BASE_DIR) if 'properties' not in dir and '.DS_Store' not in dir]
    for dir in listdir:
        if not os.path.exists('{base_dir}/{dir}/station_data.json'.format(base_dir=BASE_DIR, dir=dir)):
            listdir.remove(dir)

    '''
    inputs to consider:
        - From data: Height (DSM), ..
        - From Station: Julian Day, Day Time, Habitat, Wind Speed, Air Temperature, Ground Temperature Humidity, Pressure, Radiation.   
    '''

    inputs = len(data_map.keys()) + len(IRMaker.STATION_PARAMS_TO_USE)
    if sampling_method == 'SPP':
        X, y = pixel_to_pixel_sampling(num_samples, inputs, listdir, 'Simple')
    if sampling_method == 'RPP':
        X, y = pixel_to_pixel_sampling(num_samples, inputs, listdir, 'Relative')
    if sampling_method == 'SFP':
        X, y = frame_to_pixel_sampling(num_samples, inputs, listdir, 'Simple')
    if sampling_method == 'RFP':
        X, y = frame_to_pixel_sampling(num_samples, inputs, listdir, 'Relative')

    return X, y


def get_frame(image, i, j, radius):
    return image[i - radius: i + radius + 1, j - radius: j + radius + 1]
