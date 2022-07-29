import numpy as np


def rotate_90(frame, p):
    return np.rot90(frame, axes=(1, 2))


def rotate_180(frame, p):
    return np.rot90(frame, 2, axes=(1, 2))


def rotate_270(frame, p):
    return np.rot90(frame, 3, axes=(1, 2))


def vflip(frame, p):
    return np.flip(frame, axis=1)


def hflip(frame, p):
    return np.flip(frame, axis=2)


def noise(frame, p):
    weights = [1-p, p]
    mask = np.random.choice([0, 1], p=weights, size=frame.flatten().shape[0]).reshape(frame.shape)
    noise = np.random.uniform(0, 1, frame.flatten().shape[0]).reshape(frame.shape)
    return frame * (1 - mask) + mask * noise
