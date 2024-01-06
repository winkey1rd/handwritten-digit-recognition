import json
import os.path

import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from configuration.literals import *
from tensorflow_models import cuda_config
from tensorflow_models.architecture import compile_model, get_cnn_model
from dataset import unzip_data, normalize_data
from tensorflow_models.training import train_model

with open('configuration/config.json') as config_file:
    config = json.load(config_file)


def main():
    zip_file, train_file = [config[DATA][i] for i in (ZIP, TRAIN)]
    # create dataset
    input_shape = tuple(config[DATASET][INPUT])
    if not os.path.exists(train_file):
        if os.path.exists(zip_file):
            unzip_data(zip_file)
        else:
            print(f'Invalid train files {zip_file}, {train_file}')
            return
    try:
        train = np.loadtxt(train_file, skiprows=1, delimiter=',')
    except Exception as e:
        print(f'Error with load train file {train_file}: {e}')
        return

    try:
        x_train = normalize_data(train[:, 1:], input_shape)
        y_train = to_categorical(train[:, 0])
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=config[DATASET][VAL],
                                                          random_state=config[FIT][SEED])
    except Exception as e:
        print(f'Error with creating dataset: {e}')
        return

    try:
        comp_params = (LOSS, OPTIMIZER, METRICS)
        model = compile_model(get_cnn_model, input_shape, config[DATASET][CLASSES],
                              *[config[FIT][i] for i in comp_params])
    except Exception as e:
        print(f'Error with model compile: {e}')
        return

    try:
        train_model((x_train, y_train), (x_val, y_val), model, config[AUGMENT], config[CHECKPOINT],
                    config[REDUCE], config[STOPPING], config[TENSORBOARD], config[FIT])
    except Exception as e:
        print(f'Error with model train: {e}')
        return


if __name__ == '__main__':
    cuda_config.configurate()
    main()
