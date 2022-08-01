import json

import numpy as np
import tifffile as tiff
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import cm

from Dataset import Dataset
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IRMaker(object):
    # STATION_PARAMS_TO_USE = ["julian_day", 'time', "habitat", "wind_speed", "temperature", "humidity", "pressure",
    #                          "radiation",
    #                          "IR_temp"]
    STATION_PARAMS_TO_USE = []#["julian_day", "time", "habitat"]
    STATION_PARAMS_COUNT = len(STATION_PARAMS_TO_USE)
    DATA_MAPS = ['Height', 'RealSolar', 'Shade', 'SkyView', 'SLP', 'TGI']
    DATA_MAPS_COUNT = len(DATA_MAPS)
    FRAME_RADIUS = 12
    FRAME_WINDOW = FRAME_RADIUS * 2 + 1

    def __init__(self, dir, opt):
        super(IRMaker, self).__init__()
        self.dir = dir
        self.Height = tiff.imread('{base_dir}/{dir}/height.tif'.format(base_dir=BASE_DIR, dir=dir))
        self.RealSolar = tiff.imread('{base_dir}/{dir}/real_solar.tif'.format(base_dir=BASE_DIR, dir=dir))
        self.Shade = tiff.imread('{base_dir}/{dir}/shade.tif'.format(base_dir=BASE_DIR, dir=dir))
        self.SkyView = tiff.imread('{base_dir}/{dir}/skyview.tiff'.format(base_dir=BASE_DIR, dir=dir))
        self.SLP = tiff.imread('{base_dir}/{dir}/SLP.tif'.format(base_dir=BASE_DIR, dir=dir))
        self.TGI = tiff.imread('{base_dir}/{dir}/TGI.tif'.format(base_dir=BASE_DIR, dir=dir))
        self.RGB = tiff.imread('{base_dir}/{dir}/RGB.tif'.format(base_dir=BASE_DIR, dir=dir))[:, :, :3]

        if opt['normalize']:
            self.Height = (self.Height + 1) / 100
            self.SkyView = (self.SkyView + 1) / 3
            self.SLP = (self.SLP + 1) / 3
            self.TGI = (self.TGI + 1) / 3

        self.IR = None
        if opt['to_train']:
            self.IR = tiff.imread('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=dir))
            if opt['isCE']:
                self.IR = self.IR * IR_TEMP_FACTOR
            elif opt['normalize']:
                self.IR = self.IR / (TEMP_SCALE * IR_TEMP_FACTOR)
            if opt['bias'] is None:
                self.IR -= np.mean(self.IR)
            else:
                self.IR -= opt['bias']
            self.mu = np.mean(self.IR)
            self.sigma = np.sqrt(np.average(np.power(self.IR, 2)))

            if opt['isCE']:
                self.IR = (self.IR + POSITIVE_CONST).astype(int)
                if opt['use_loss_weights']:
                    loss_weights = 1 / np.cbrt((np.bincount(self.IR.flatten()) + 1))
                    loss_weights = (loss_weights < 1) * loss_weights
                    if loss_weights.shape[0] < TEMP_SCALE * IR_TEMP_FACTOR:
                        self.loss_weights = np.concatenate((loss_weights, np.zeros(TEMP_SCALE * IR_TEMP_FACTOR - loss_weights.shape[0])))
                    else:
                        self.loss_weights = loss_weights[: TEMP_SCALE * IR_TEMP_FACTOR]
                    self.loss_weights /= np.sum(self.loss_weights)

                else:
                    self.loss_weights = None

            else:
                self.loss_weights = None

        with open('{base_dir}/{dir}/station_data.json'.format(base_dir=BASE_DIR, dir=dir), 'r') as f:
            self.station_data = json.loads(f.read())

        self.RealSolar = (np.average(self.RealSolar[1:-1, 1:-1]) * (self.RealSolar < 0) * 1. + \
                                self.RealSolar * (self.RealSolar >= 0) * 1.) / 1500

    def generate_image(self, model):
        model.eval()
        if model.name == 'ConvNet':
            predicted_IR = self.generate_image_conv(model)
        else:
            predicted_IR = np.zeros_like(self.Height)

            inputs = self.DATA_MAPS_COUNT + self.STATION_PARAMS_COUNT
            image_pixels = self.Height.shape[0] * self.Height.shape[1]
            X = np.zeros(shape=(image_pixels, inputs), dtype=np.float)
            batch_size = image_pixels // 10

            X[:, 0] = self.Height.reshape(1, -1)
            X[:, 1] = self.RealSolar.reshape(1, -1)
            X[:, 2] = self.Shade.reshape(1, -1)
            X[:, 3] = self.SkyView.reshape(1, -1)
            X[:, 4] = self.SLP.reshape(1, -1)
            X[:, 5] = self.TGI.reshape(1, -1)

            for i, key in enumerate(self.STATION_PARAMS_TO_USE):
                X[:, 6 + i] = np.tile(self.station_data[key], image_pixels)

            dataset = Dataset(X, np.zeros(shape=X.shape[0]))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            i = 0
            j = 0
            for x, y in data_loader:
                x, y, *data = x.float(), y.float()
                y_hat = model(x)
                pred = model.predict(y_hat)
                for pixel in pred:
                    predicted_IR[i][j] = pixel
                    j += 1
                    if j >= predicted_IR.shape[1]:
                        j = 0
                        i += 1
        predicted_IR = predicted_IR / IR_TEMP_FACTOR
        tiff.imsave('{base_dir}/{dir}/PredictedIR.tif'.format(base_dir=BASE_DIR, dir=self.dir), predicted_IR)
        tiff.imsave('{base_dir}/{dir}/PredictedIRGrayscale.tif'.format(base_dir=BASE_DIR, dir=self.dir),
                    self.normalize_image(predicted_IR))
        self.IR = tiff.imread('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=self.dir))
        tiff.imsave('{base_dir}/{dir}/IRGrayscale.tif'.format(base_dir=BASE_DIR, dir=self.dir),
                    self.normalize_image(self.IR))
        evaluate_prediceted_IR(self.dir)

    def generate_image_conv(self, model):  # TODO merge with original function
        predicted_IR = np.zeros_like(self.Height)
        inputs = self.DATA_MAPS_COUNT + self.STATION_PARAMS_COUNT
        image_pixels = self.Height.shape[0] * self.Height.shape[1]
        batch_size = image_pixels // 10
        dir_data = self.get_data_dict()
        station_data = self.station_data
        dir_data = [np.pad(image, IRMaker.FRAME_RADIUS, 'constant') for image in dir_data]
        # for i in range(self.Height.shape[0]):
        #     for j in range(self.Height.shape[1]):
        for i in range(self.Height.shape[0]):
            if i % 10 == 0:
                print('{}/{}'.format(i, self.Height.shape[0]))
            for j in range(self.Height.shape[1]):
                data_samples = list()
                for image in dir_data:
                    flat_image = get_frame(image, i + IRMaker.FRAME_RADIUS, j + IRMaker.FRAME_RADIUS, IRMaker.FRAME_RADIUS).flatten()
                    data_samples.extend(flat_image)
                for key in self.STATION_PARAMS_TO_USE:
                    data_samples.append(station_data[key])
                X, data = torch.Tensor(data_samples)[:3750].to(device), torch.Tensor(data_samples)[3750:].to(device)
                data = data.reshape(1, data.size()[0])
                X = X.reshape((1, 6, 25, 25))  # TODO replace with params
                y_hat = model(X, data)
                predicted_IR[i][j] = model.predict(y_hat)
        return predicted_IR

    def generate_error_images(self):
        cmap_range = 5
        cmap_mask = (np.arange(1, 257) * (np.arange(1, 257) > 128))
        cmap_mask = (cmap_mask % (128 // cmap_range - 1) == 0) & (cmap_mask != 0)
        cmap_mask[255] = 1 if np.sum(cmap_mask) < cmap_range else 0

        cmaps = [(cm.Blues(range(256))[:, :3][cmap_mask] * 255).astype(np.uint8),
                 (cm.Reds(range(256))[:, :3][cmap_mask] * 255).astype(np.uint8),
                 np.array([[247, 225, 75]], dtype=np.uint8),
                 (cm.Oranges(range(256))[:, :3][cmap_mask] * 255).astype(np.uint8),
                 (cm.Purples(range(256))[:, :3][cmap_mask] * 255).astype(np.uint8),
                 (cm.Greens(range(256))[:, :3][cmap_mask] * 255).astype(np.uint8)]
        names = ['Height', 'RealSolar', 'Shade', 'SkyView', 'SLP', 'TGI']

        color_images = zip(self.get_data_dict(), cmaps, names)
        predicted_IR = tiff.imread('{base_dir}/{dir}/PredictedIR.tif'.format(base_dir=BASE_DIR, dir=self.dir))
        IR = tiff.imread('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=self.dir))
        norm_predicted_IR = IRMaker.normalize_image(np.repeat(np.expand_dims(predicted_IR, 2), 3, 2))

        error_image = np.abs(predicted_IR - IR) >= 2
        error_image = np.repeat(np.expand_dims(error_image, 2), 3, 2)
        for image, cmap, name in color_images:
            norm_image = IRMaker.normalize_image(image)
            min = np.min(norm_image)
            max = np.max(norm_image)
            rgb_image = np.repeat(np.expand_dims(norm_image, 2), 3, 2)
            color_image = np.zeros_like(rgb_image, dtype=np.uint8)
            for i in range(cmap.shape[0]):
                low_level = min + ((max - min) / cmap.shape[0]) * i
                high_level = min + ((max - min) / cmap.shape[0]) * (i + 1)
                if i == 0:
                    color_image = color_image + ((low_level <= rgb_image) & (rgb_image <= high_level)) * cmap[i]
                else:
                    color_image = color_image + ((low_level < rgb_image) & (rgb_image <= high_level)) * cmap[i]

            color_image = error_image * color_image + (error_image == 0) * rgb_image
            color_predicted_IR = error_image * color_image + (error_image == 0) * norm_predicted_IR
            tiff.imsave('{base_dir}/{dir}/Error_{name}.tif'.format(base_dir=BASE_DIR, dir=self.dir, name=name), color_image)
            tiff.imsave('{base_dir}/{dir}/Error_{name}_on_PredictedIR.tif'.format(base_dir=BASE_DIR, dir=self.dir, name=name), color_predicted_IR)

    def get_data_dict(self):
        return [self.Height, self.RealSolar, self.Shade, self.SkyView, self.SLP, self.TGI]

    @staticmethod
    def normalize_image(image):
        image = image - np.min(image)
        image = (image / np.max(image)) * 255
        return image.astype(np.uint8)


# TODO temp
def get_frame(image, i, j, radius):
    return image[i - radius: i + radius + 1, j - radius: j + radius + 1]
