import json

import numpy as np
import tifffile as tiff
from collections import OrderedDict

from torch.utils.data import DataLoader

from Dataset import Dataset

BASE_DIR = './resources'
STATION_PARAMS_TO_USE = ["julian_day", 'time', "habitat", "wind_speed", "temperature", "humidity", "pressure", "radiation",
                         "IR_temp"]

data_map = OrderedDict({
    'Height': None,
    'RealSolar': None,
    'Shade': None,
    'SkyView': None,
    'SLP': None,
    'TGI': None,
})

class IRMaker(object):
    def __init__(self, dir, train):
        super(IRMaker, self).__init__()
        self.dir = dir

        self.Height = tiff.imread('{base_dir}/{dir}/height.tif'.format(base_dir=BASE_DIR, dir=dir))
        self.RealSolar = tiff.imread('{base_dir}/{dir}/real_solar.tif'.format(base_dir=BASE_DIR, dir=dir))
        self.Shade = tiff.imread('{base_dir}/{dir}/shade.tif'.format(base_dir=BASE_DIR, dir=dir))
        self.SkyView = tiff.imread('{base_dir}/{dir}/skyview.tiff'.format(base_dir=BASE_DIR, dir=dir))
        self.SLP = tiff.imread('{base_dir}/{dir}/SLP.tif'.format(base_dir=BASE_DIR, dir=dir))
        self.TGI = tiff.imread('{base_dir}/{dir}/TGI.tif'.format(base_dir=BASE_DIR, dir=dir))

        self.IR = None
        if train:
            self.IR = tiff.imread('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=dir)) + 273.15

        with open('{base_dir}/{dir}/station_data.json'.format(base_dir=BASE_DIR, dir=dir), 'r') as f:
            self.station_data = json.loads(f.read())

        self.RealSolar = np.average(self.RealSolar[1:-1, 1:-1]) * (self.RealSolar < 0) * 1. + \
                                self.RealSolar * (self.RealSolar >= 0) * 1.


    def generate_image(self, model):
        model.eval()
        predicted_IR = np.zeros_like(self.Height)

        inputs = len(data_map.keys()) + len(STATION_PARAMS_TO_USE)
        image_pixels = self.Height.shape[0] * self.Height.shape[1]
        X = np.zeros(shape=(image_pixels, inputs), dtype=np.float)
        batch_size = image_pixels // 10

        X[:, 0] = self.Height.reshape(1, -1)
        X[:, 1] = self.RealSolar.reshape(1, -1)
        X[:, 2] = self.Shade.reshape(1, -1)
        X[:, 3] = self.SkyView.reshape(1, -1)
        X[:, 4] = self.SLP.reshape(1, -1)
        X[:, 5] = self.TGI.reshape(1, -1)

        for i, key in enumerate(STATION_PARAMS_TO_USE):
            X[:, 6 + i] = np.tile(self.station_data[key], image_pixels)

        dataset = Dataset(X, np.zeros(shape=X.shape[0]))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for x, y in data_loader:
            x, y = x.float(), y.float()
            y_hat = model(x).detach()
            pred = np.array(y_hat.detach())








