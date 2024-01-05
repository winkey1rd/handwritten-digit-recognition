import zipfile
import os

import numpy as np


def load_data(zip_path: str, train_name: str, test_name: str):
    """
    func get train and test data from zip
    :param zip_path: path to zip
    :param train_name: filename in zip with train data
    :param test_name: filename in zip with test data
    :return: train, test numpy arrays
    """
    path = os.path.dirname(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall()
    train = np.loadtxt(f'{path}/{train_name}', skiprows=1, delimiter=',')
    test = np.loadtxt(f'{path}/{test_name}', skiprows=1, delimiter=',')
    return train, test


def normalize_data(x, input_shape):
    x = x.reshape(x.shape[0], *input_shape)
    x /= 255.0
    return x
