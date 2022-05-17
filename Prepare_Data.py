import json
import os

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


def prepare_data(samples, dir=None):
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
    X = np.zeros(shape=(samples * len(listdir), inputs), dtype=np.float)
    y = np.zeros(shape=(samples * len(listdir)), dtype=np.float)
    k = 0

    for dir in listdir:
        IRObj = IRMaker(dir, train=True)
        dir_data = IRObj.get_data_dict()
        station_data = IRObj.station_data
        label_data = IRObj.IR

        row_indices = np.random.choice(dir_data[0].shape[0], samples)
        col_indices = np.random.choice(dir_data[0].shape[0], samples)

        for i, j in zip(row_indices, col_indices):
            data_samples = list()
            for image in dir_data:
                data_samples.append(image[i][j])
            for key in IRObj.STATION_PARAMS_TO_USE:
                data_samples.append(station_data[key])
            X[k] = np.array(data_samples)
            y[k] = label_data[i][j]
            k += 1

    return X, y