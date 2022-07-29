import random
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch

from IRMaker import IRMaker
from Main import get_best_model, device
from Prepare_Data import get_frame
from utils import *
from Dataset import Dataset
from Models import *


ALLOW_CLOSE_PIXELS = False

def test_model(model):
    model.eval()
    results = list()
    for x, y in model.valid_loader:
        x, y, *data = model.unpack((x, y), device)
        y_hat = model(x) if not data else model(x, data[0])
        results.append(tuple(model.predict(y_hat).tolist()))
    return results


def evaluate_similarity_by_metric(first_indices, second_indices, image, metric):
    i, j = first_indices
    m, n = second_indices
    if metric == 'mae':
        return np.average(np.abs(get_frame(image, i, j) - get_frame(image, m, n)))
    if metric == 'mse':
        return np.average(np.power(get_frame(image, i, j) - get_frame(image, m, n), 2))
    if metric == 'temp_env':
        return (IRMaker.FRAME_WINDOW ** 2 - np.sum((get_frame(image, m, n) < get_frame(image, i, j) + DEGREE_ERROR) * \
               (get_frame(image, m, n) > get_frame(image, i, j) - DEGREE_ERROR))) / (IRMaker.FRAME_WINDOW ** 2 * 0.9)


def random_similar_sampling_by_method(method, image, metric):
    if method == 'Relative':
        low = np.min(image)
        high = np.max(image)
        num_level_sample = int((high - low) / (IR_TEMP_DIFF / (TEMP_SCALE * IR_TEMP_FACTOR)))

        levels = [list() for _ in range(num_level_sample + 1)]
        for i in range(image.shape[0]):
            for j in range(image.shape[0]):
                levels[int((image[i][j] - low) // (IR_TEMP_DIFF / (TEMP_SCALE * IR_TEMP_FACTOR)))].append((i, j))

        test_indices = list()
        pad_image = np.pad(image, IRMaker.FRAME_RADIUS)
        for i in range(len(levels)):
            if len(levels[i]) < 2:
                continue
            FIRST_TRY = True
            min_score = (high - low) * IRMaker.FRAME_WINDOW ** 2
            base_indices = random.sample(levels[i], 1)[0]
            test_indices.append([base_indices, None])
            levels[i].remove(base_indices)
            for j in range(2):
                for indices in levels[i]:
                    if np.sqrt(np.power(base_indices[0] - indices[0], 2) + np.power(base_indices[1] - indices[1], 2)) < IRMaker.FRAME_RADIUS and\
                            FIRST_TRY:
                        continue
                    score = evaluate_similarity_by_metric(base_indices, indices, pad_image, metric)
                    if score < min_score and score < SIMILARITY_THRESHOLD:
                        test_indices[-1][-1] = indices
                        min_score = score
                    # if score < SIMILARITY_THRESHOLD:
                    #     break
                if test_indices[-1][-1] is None:
                    if not ALLOW_CLOSE_PIXELS:
                        test_indices.pop(-1)
                        break
                    FIRST_TRY = False


        test_row, test_col = zip(*test_indices)
        return test_row, test_col

def frame_to_pixel_similarities(dir, method, metric):
    IRObj = IRMaker(dir, opt)
    dir_data = [np.pad(image, IRMaker.FRAME_RADIUS) for image in IRObj.get_data_dict()]
    station_data = IRObj.station_data
    label_data = IRObj.IR

    low = np.min(label_data)
    high = np.max(label_data)
    num_samples = 2 * int((high - low) / (IR_TEMP_DIFF / (TEMP_SCALE * IR_TEMP_FACTOR)))
    X = np.zeros(shape=(int(num_samples) * 2, IRMaker.DATA_MAPS_COUNT * (IRMaker.FRAME_WINDOW ** 2) + IRMaker.STATION_PARAMS_COUNT), dtype=float)
    y = np.zeros(shape=(int(num_samples)) * 2, dtype=float)
    k = 0

    row, col = random_similar_sampling_by_method(method, IRObj.IR, metric)

    for i_pair, j_pair in zip(row, col):
        for l in range(2):
            data_samples = list()
            for image in dir_data:
                frame = get_frame(image, i_pair[l], j_pair[l]).flatten()
                data_samples.extend(frame)
            for key in IRObj.STATION_PARAMS_TO_USE:
                data_samples.append(station_data[key])
            X[k] = np.array(data_samples)
            y[k] = label_data[i_pair[l]][j_pair[l]]
            k += 1

    return X[:k], y[:k]


def find_similar_frames(model_name, dir):
    # Create json station data for each folder
    csv_to_json('./resources/properties/data_table.csv')

    if model_name == 'InceptionV3':
        IRMaker.FRAME_RADIUS, IRMaker.FRAME_WINDOW = 149, 299
    if model_name == 'VGG19':
        IRMaker.FRAME_RADIUS, IRMaker.FRAME_WINDOW = 112, 224

    return frame_to_pixel_similarities(dir, 'Relative', 'temp_env')


def find_all_frames(num_samples, dir):
    IRObj = IRMaker(dir, opt)
    dir_data = [np.pad(image, IRMaker.FRAME_RADIUS) for image in IRObj.get_data_dict()]
    station_data = IRObj.station_data
    label_data = IRObj.IR

    row_indices = np.random.choice(label_data.shape[0], num_samples)
    col_indices = np.random.choice(label_data.shape[0], num_samples)
    indices = list(zip(row_indices, col_indices))
    random.shuffle(indices)

    X = np.zeros(shape=(num_samples, IRMaker.DATA_MAPS_COUNT * (IRMaker.FRAME_WINDOW ** 2) + IRMaker.STATION_PARAMS_COUNT),
                 dtype=float)
    y = np.zeros(shape=(num_samples), dtype=float)
    k = 0

    for i, j in indices:
        data_samples = list()
        for image in dir_data:
            frame = get_frame(image, i, j).flatten()
            data_samples.extend(frame)
        for key in IRObj.STATION_PARAMS_TO_USE:
            data_samples.append(station_data[key])
        X[k] = np.array(data_samples)
        y[k] = label_data[i][j]
        k += 1

    return X[:k], y[:k]

def update_model_for_testing(model, dir):
    X, y = find_similar_frames(model.name, dir)
    batch_size = 2
    ds = Dataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size)
    model.valid_loader = dl


if __name__ == '__main__':
    # Choose Model: 'IRValue', 'IRClass', 'ConvNet', 'ResNet18', 'ResNet50', 'InceptionV3', 'VGG19', 'ResNetXt101'
    dir = 'Zeelim_7.11.19_1550_W'
    model_name = 'ConvNet'
    # X, y = find_similar_frames('ConvNet', dir)
    X, y = find_all_frames(1000, dir)
    count = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            X_diff = np.average(np.abs(X[i] - X[j]))
            y_diff = np.abs(y[i] - y[j]) * 210
            if (X_diff < 0.05 and y_diff > 1):
                count += 1
    print('Sampled Error is: {}'.format(np.round(count / X.shape[0] ** 2, 4)))
    print(1)
    # model = get_best_model(model_name)
    # if not model:
    #     print("There's no save model. Create and train model first through Main.py")
    #     exit(0)
    # update_model_for_testing(model, dir)
    # results = test_model(model)
    # for pair in results:
    #     print(pair)
