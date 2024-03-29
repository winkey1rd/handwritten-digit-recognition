import json
from multiprocessing import cpu_count

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import config as tensorflow_config


from configuration.literals import *
from models.architecture import compile_model, get_cnn_model
from dataset import load_data, normalize_data
from models.training import train_model

with open('configuration/config.json') as config_file:
    config = json.load(config_file)


def main():
    zip_file = "../data/digit-recognizer.zip"
    train_file = "train.csv"
    test_file = "test.csv"

    # create dataset
    input_shape = tuple(config[DATASET][INPUT])
    train, test = load_data(zip_file, train_file, test_file)
    x_train = normalize_data(train[:, 1:], input_shape)
    y_train = to_categorical(train[:, 0])

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=config[DATASET][VAL],
                                                      random_state=config[FIT][SEED])

    loss = "categorical_crossentropy"
    optimizer = 'adam'
    metrics = ['accuracy']
    model = compile_model(get_cnn_model, input_shape, config[DATASET][CLASSES], loss, optimizer, metrics)

    train_model((x_train, y_train), (x_val, y_val), model, config[AUGMENT], config[CHECKPOINT],
                config[REDUCE], config[STOPPING], config[TENSORBOARD], config[FIT])


if __name__ == '__main__':
    num_cores = min(max(config[SOFT][WORKER], 1), max(cpu_count() - 1, 1))
    tensorflow_config.threading.set_inter_op_parallelism_threads(num_cores)
    tensorflow_config.threading.set_intra_op_parallelism_threads(num_cores)

    physical_gpu = tensorflow_config.list_physical_devices('GPU')
    physical_cpu = tensorflow_config.list_physical_devices('CPU')

    gpu_count = min(len(physical_gpu), len(config[SOFT][GPU]))
    gpu_devices = [physical_gpu[i] for i in config[SOFT][GPU]]
    mem_limit = False if gpu_count == 0 else config[SOFT][GROWTH]
    tensorflow_config.set_visible_devices(gpu_devices, 'GPU')
    for device in gpu_devices:
        try:
            tensorflow_config.experimental.set_memory_growth(device, mem_limit)
        except Exception as e:
            print(f'Error tensorflow memory configuration: {e}')
    main()
