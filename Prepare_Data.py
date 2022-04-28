import json
import os
from PIL import Image
from collections import OrderedDict
import numpy as np


BASE_DIR = './resources'
STATION_PARAMS_TO_USE = ["julian_day", "habitat", "wind_speed", "temperature", "humidity", "pressure", "radiation",
                         "IR_temp"]

data_map = OrderedDict({
    'Height': None,
    'IR': None,
    'RealSolar': None,
    'Shade': None,
    'SkyView': None,
    'SLP': None,
    'TGI': None,
})
listdir = [dir for dir in os.listdir(BASE_DIR) if 'properties' not in dir]

'''
    inputs to consider:
        - From data: Height (DSM), ..
        - From Metro-Station: Julian Day, Day Time, Habitat, Wind Speed, Air Temperature, Ground Temperature Humidity, Pressure, Radiation.   
'''

inputs = len(data_map.keys()) + len(STATION_PARAMS_TO_USE)
samples = 1000
X = np.zeros(shape=(samples * len(listdir), inputs))

for dir in listdir:
    dir_data = data_map.copy()
    dir_data['Height'] = np.array(Image.open('{base_dir}/{dir}/height.tif'.format(base_dir=BASE_DIR, dir=dir)))
    dir_data['IR'] = np.array(Image.open('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=dir)))
    dir_data['RealSolar'] = np.array(Image.open('{base_dir}/{dir}/real_solar.tif'.format(base_dir=BASE_DIR, dir=dir)))
    dir_data['Shade'] = np.array(Image.open('{base_dir}/{dir}/shade.tif'.format(base_dir=BASE_DIR, dir=dir)))
    dir_data['SkyView'] = np.array(Image.open('{base_dir}/{dir}/sky_view.tif'.format(base_dir=BASE_DIR, dir=dir)))
    dir_data['SLP'] = np.array(Image.open('{base_dir}/{dir}/SLP.tif'.format(base_dir=BASE_DIR, dir=dir)))
    dir_data['TGI'] = np.array(Image.open('{base_dir}/{dir}/TGI.tif'.format(base_dir=BASE_DIR, dir=dir)))

    metro_station = json.loads('{base_dir}/{dir}/station_data.json'.format(base_dir=BASE_DIR, dir=dir))

    row_indices = np.random.choice(1000, samples)
    col_indices = np.random.choice(1000, samples)

    for k, i, j in enumerate(zip(row_indices, col_indices)):
        data_samples = list()
        for image in dir_data.values():
            data_samples.append(image[i][j])
        X[k] = np.array(data_samples)

print(1)


def generate_data(samples, dir=None):
    listdir = [dir] if dir else [dir for dir in os.listdir(BASE_DIR) if 'data_table' not in dir]

def dir_to_dataset(dir):

