import json
import os

from keras.saving.save import load_model
import numpy as np

from configuration.literals import *
from tensorflow_models import cuda_config

from dataset import unzip_data, normalize_data

with open('configuration/config.json') as config_file:
    config = json.load(config_file)


def main():
    zip_file, test_file = [config[DATA][i] for i in (ZIP, TEST)]
    # create dataset
    input_shape = tuple(config[DATASET][INPUT])
    if not os.path.exists(test_file):
        if os.path.exists(zip_file):
            unzip_data(zip_file)
        else:
            print(f'Invalid train files {zip_file}, {test_file}')
            return
    try:
        test = np.loadtxt(test_file, skiprows=1, delimiter=',')
    except Exception as e:
        print(f'Error with load test file {test_file}: {e}')
        return

    try:
        x_test = normalize_data(test, input_shape)
    except Exception as e:
        print(f'Error with processing test data: {e}')
        return

    try:
        model = load_model(config[FIT][NAME])
    except Exception as e:
        print(f'Error with load model: {e}')
        return

    try:
        predictions = model.predict(x_test)
        predictions = np.argmax(predictions, axis=1)
    except Exception as e:
        print(f'Error with model predict: {e}')
        return

    try:
        header = 'ImageId,Label'
        out = np.column_stack((range(1, predictions.shape[0] + 1), predictions))
        np.savetxt(config[DATA][OUT], out, header=header,  comments='', fmt='%d,%d')
    except Exception as e:
        print(f'Error with saving prediction results: {e}')


if __name__ == '__main__':
    cuda_config.configurate()
    main()
