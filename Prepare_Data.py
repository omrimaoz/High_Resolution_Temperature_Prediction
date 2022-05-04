import json
import os

import tifffile as tiff
from PIL import Image
from collections import OrderedDict
import numpy as np
from IRMaker import IRMaker


BASE_DIR = './resources'

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


# def create_data(dir):
#     dir_data = data_map.copy()
#     dir_data['Height'] = tiff.imread('{base_dir}/{dir}/height.tif'.format(base_dir=BASE_DIR, dir=dir))
#     dir_data['RealSolar'] = tiff.imread('{base_dir}/{dir}/real_solar.tif'.format(base_dir=BASE_DIR, dir=dir))
#     dir_data['Shade'] = tiff.imread('{base_dir}/{dir}/shade.tif'.format(base_dir=BASE_DIR, dir=dir))
#     dir_data['SkyView'] = tiff.imread('{base_dir}/{dir}/skyview.tiff'.format(base_dir=BASE_DIR, dir=dir))
#     dir_data['SLP'] = tiff.imread('{base_dir}/{dir}/SLP.tif'.format(base_dir=BASE_DIR, dir=dir))
#     dir_data['TGI'] = tiff.imread('{base_dir}/{dir}/TGI.tif'.format(base_dir=BASE_DIR, dir=dir))
#
#     label_data = tiff.imread('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=dir)) + 273.15
#
#     with open('{base_dir}/{dir}/station_data.json'.format(base_dir=BASE_DIR, dir=dir), 'r') as f:
#         station_data = json.loads(f.read())
#
#     dir_data['RealSolar'] = np.average(dir_data['RealSolar'][1:-1, 1:-1]) * (dir_data['RealSolar'] < 0) * 1. +\
#                         dir_data['RealSolar'] * (dir_data['RealSolar'] >= 0) * 1.
#     return dir_data, station_data, label_data

