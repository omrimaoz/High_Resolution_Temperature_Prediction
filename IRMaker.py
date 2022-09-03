import json
import time

import numpy as np
import tifffile as tiff
from PIL import Image
from matplotlib.colors import ListedColormap
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import cm

from Dataset import Dataset
from utils import *


class IRMaker(object):
    # STATION_PARAMS_TO_USE = ["julian_day", 'time', "habitat", "wind_speed", "temperature", "humidity", "pressure",
    #                          "radiation",
    #                          "IR_temp"]
    STATION_PARAMS_TO_USE = []#["julian_day", "time", "habitat"]
    STATION_PARAMS_COUNT = len(STATION_PARAMS_TO_USE)
    # DATA_MAPS = ['Height', 'RealSolar', 'SLP', 'SkyView', 'Shade', 'TGI']
    DATA_MAPS = ['Height', 'RealSolar', 'SLP']
    # DATA_MAPS = ['Height', 'SkyView', 'SLP']
    # DATA_MAPS = ['RGB_R', 'RGB_G', 'RGB_B']
    DATA_MAPS_COUNT = len(DATA_MAPS)
    FRAME_RADIUS = 15
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
        self.RGB_R = self.RGB[:, :, 0]
        self.RGB_G = self.RGB[:, :, 1]
        self.RGB_B = self.RGB[:, :, 2]

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
                self.mean = np.mean(self.IR)
                self.IR -= self.mean
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

    def generate_image(self, opt):
        start = time.time()
        opt['model'].eval()
        if opt['model'].name in ['ConvNet', 'ResNet18', 'ResNet50', 'InceptionV3', 'VGG19', 'ResNetXt101']:
            predicted_IR = self.generate_image_conv(opt)
        else:
            predicted_IR = self.generate_image_simple(opt)
        print('Generated PredictedIR Image Time: {}'.format(time.time() - start))
        tiff.imsave('{base_dir}/{dir}/PredictedIR.tif'.format(base_dir=BASE_DIR, dir=self.dir), predicted_IR)
        tiff.imsave('{base_dir}/{dir}/PredictedIRGrayscale.tif'.format(base_dir=BASE_DIR, dir=self.dir),
                    self.normalize_image(predicted_IR))
        self.IR = tiff.imread('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=self.dir))
        tiff.imsave('{base_dir}/{dir}/IRGrayscale.tif'.format(base_dir=BASE_DIR, dir=self.dir),
                    self.normalize_image(self.IR))
        evaluate_prediceted_IR(self.dir, opt)

    def generate_image_simple(self, opt):
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
            y_hat = opt['model'](x)
            pred = opt['model'].predict(y_hat)
            for pixel in pred:
                predicted_IR[i][j] = pixel
                j += 1
                if j >= predicted_IR.shape[1]:
                    j = 0
                    i += 1

        predicted_IR /= IR_TEMP_FACTOR
        return predicted_IR

    def generate_image_conv(self, opt):
        predicted_IR = np.zeros_like(self.Height)
        dtype = np.int if opt['isCE'] else np.float
        image_pixels = self.Height.shape[1]
        input_image_num = IRMaker.DATA_MAPS_COUNT if opt['label_kind'] == 'ir' else 3
        s = 0
        start = time.time()
        for i in range(self.Height.shape[0]):
            X = np.zeros(shape=(image_pixels, input_image_num * (IRMaker.FRAME_WINDOW ** 2) + IRMaker.STATION_PARAMS_COUNT), dtype=np.float)
            y = np.arange(image_pixels, dtype=dtype)
            k = 0

            batch_size = self.Height.shape[0]
            station_data = self.station_data
            dir_data = [np.pad(image, IRMaker.FRAME_RADIUS) for image in self.get_data_dict()]
            if (i+1) % 10 == 0:
                print('{}/{} time:{}'.format(i+1, self.Height.shape[0], (time.time() - start) / 60))
            for j in range(self.Height.shape[1]):
                data_samples = list()
                for image in dir_data:
                    flat_image = self.get_frame(image, i, j).flatten()
                    data_samples.extend(flat_image)
                for key in self.STATION_PARAMS_TO_USE:
                    data_samples.append(station_data[key])
                X[k] = np.array(data_samples)
                k += 1

            ds = Dataset(X, y)
            dl = DataLoader(ds, batch_size=batch_size)

            for t, pack in enumerate(dl):
                x, y, *data = opt['model'].unpack(pack, device)
                y_hat = opt['model'](x) if not data else opt['model'](x, data[0])
                y_hat = opt['model'].predict(y_hat)
                predicted_IR[s] = y_hat.cpu().numpy()
                s += 1

        predicted_IR = self.reverse_normalization(predicted_IR, opt)
        return predicted_IR

    def reverse_normalization(self, predicted_IR, opt):
        if opt['isCE']:
            predicted_IR = predicted_IR - POSITIVE_CONST
        if opt['bias'] is None:
            predicted_IR += self.mean
        else:
            predicted_IR += opt['bias']
        if opt['isCE']:
            predicted_IR /= IR_TEMP_FACTOR
        elif opt['normalize']:
            predicted_IR *= (TEMP_SCALE * IR_TEMP_FACTOR)

        return predicted_IR

    def generate_error_images_discrete(self, opt, num_of_levels):
        cmap_mask = (np.arange(1, 257) * (np.arange(1, 257) > 128))
        cmap_mask = (cmap_mask % (128 // num_of_levels - 1) == 0) & (cmap_mask != 0)
        cmap_mask[255] = 1 if np.sum(cmap_mask) < num_of_levels else 0

        cmaps = [(cm.Blues(range(256))[:, :3][cmap_mask] * 255).astype(np.uint8),
                 (cm.Reds(range(256))[:, :3][cmap_mask] * 255).astype(np.uint8),
                 (cm.Purples(range(256))[:, :3][cmap_mask] * 255).astype(np.uint8),
                 (cm.Oranges(range(256))[:, :3][cmap_mask] * 255).astype(np.uint8),
                 np.array([[247, 225, 75]], dtype=np.uint8),
                 (cm.Greens(range(256))[:, :3][cmap_mask] * 255).astype(np.uint8)]
        names = ['Height', 'RealSolar', 'SLP', 'SkyView', 'Shade', 'TGI']

        color_images = zip(self.get_data_dict(), cmaps, names)
        predicted_IR = tiff.imread('{base_dir}/{dir}/PredictedIR.tif'.format(base_dir=BASE_DIR, dir=self.dir))
        IR = tiff.imread('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=self.dir))
        norm_predicted_IR = IRMaker.normalize_image(np.repeat(np.expand_dims(predicted_IR, 2), 3, 2))

        error_image = np.abs(predicted_IR - IR) >= DEGREE_ERROR
        error_image = np.repeat(np.expand_dims(error_image, 2), 3, 2)
        error_level_counts = np.zeros(num_of_levels)
        for image, cmap, name in color_images:
            error_level_count = list()
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
                    error_level_count.append(np.sum(((low_level <= rgb_image) & (rgb_image <= high_level)) * error_image) / 3)
                else:
                    color_image = color_image + ((low_level < rgb_image) & (rgb_image <= high_level)) * cmap[i]
                    error_level_count.append(np.sum(((low_level < rgb_image) & (rgb_image <= high_level)) * error_image) / 3)

            error_level_counts += np.array(error_level_count)
            color_image = error_image * color_image + (error_image == 0) * rgb_image
            color_predicted_IR = error_image * color_image + (error_image == 0) * norm_predicted_IR
            tiff.imsave('{base_dir}/{dir}/Error_{name}.tif'.format(base_dir=BASE_DIR, dir=self.dir, name=name), color_image)
            tiff.imsave('{base_dir}/{dir}/Error_{name}_on_PredictedIR.tif'.format(base_dir=BASE_DIR, dir=self.dir, name=name), color_predicted_IR)

        opt['augmentation_by_level'] = (error_level_counts / np.sum(error_level_counts) * 5).astype(np.int)

    def generate_error_images_continuous(self):
        custom_cmaps = list()
        cmaps = ['Blues', 'Reds', 'Purples', 'Oranges', 'autumn', 'Greens']
        for cmap in cmaps:
            cmap_const = 128
            custom_cmaps.append(ListedColormap(cm.get_cmap(cmap, cmap_const)(np.linspace(0.5, 1, cmap_const)), name='Custom'+cmap))

        names = ['Height', 'RealSolar', 'SLP', 'SkyView', 'Shade', 'TGI']

        color_images = zip(self.get_data_dict(), custom_cmaps, names)
        predicted_IR = tiff.imread('{base_dir}/{dir}/PredictedIR.tif'.format(base_dir=BASE_DIR, dir=self.dir))
        IR = tiff.imread('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=self.dir))

        error_image = np.abs(predicted_IR - IR) < DEGREE_ERROR
        for image, cmap, name in color_images:
            masked = np.ma.masked_where(error_image * 1, image)
            plt.imshow(image, vmin =np.min(image), vmax=np.max(image), cmap='Greys')
            plt.imshow(masked, vmin=np.min(image), vmax=np.max(image), cmap=cmap, alpha=1)
            plt.colorbar()
            plt.show()
            plt.clf()

    def create_error_histogram(self):
        IR = tiff.imread('{base_dir}/{dir}/IR.tif'.format(base_dir=BASE_DIR, dir=self.dir))
        predicted_IR = tiff.imread('{base_dir}/{dir}/predictedIR.tif'.format(base_dir=BASE_DIR, dir=self.dir))
        error_indices = (np.abs(IR.flatten() - predicted_IR.flatten()) > DEGREE_ERROR)
        error_v = IR.flatten()[error_indices]
        border = np.average(error_v)
        bins = np.linspace(border - 2, border + 2, int((2 * border) / 0.2))
        plt.hist(IR.flatten(), bins, width=0.02, alpha=0.5, label='IR value', color='red')
        plt.hist(error_v, bins, width=0.02, alpha=0.5, label='error', color='blue')
        plt.title('IR Error Histogram\n{}'.format(self.dir))
        plt.ylabel('Counts')
        plt.xlabel('IR value')
        plt.legend(loc='upper right')
        plt.show()

    def get_data_dict(self):
        data = list()
        for map in self.DATA_MAPS:
            data.append(eval('self.{}'.format(map)))
        return data

    @staticmethod
    def normalize_image(image):
        image = image - np.min(image)
        image = (image / np.max(image)) * 255
        return image.astype(np.uint8)

    @staticmethod
    def get_frame(image, i, j):
        return image[i: i + IRMaker.FRAME_WINDOW, j: j + IRMaker.FRAME_WINDOW]
