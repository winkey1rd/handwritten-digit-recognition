import json
from multiprocessing import cpu_count

from tensorflow import config as tensorflow_config

from configuration.literals import *
with open('configuration/config.json') as config_file:
    config = json.load(config_file)


def configurate():
    num_cores = min(max(config[SOFT][WORKER], 1), max(cpu_count() - 1, 1))
    tensorflow_config.threading.set_inter_op_parallelism_threads(num_cores)
    tensorflow_config.threading.set_intra_op_parallelism_threads(num_cores)

    physical_gpu = tensorflow_config.list_physical_devices('GPU')
    gpu_count = min(len(physical_gpu), len(config[SOFT][GPU]))
    gpu_devices = [physical_gpu[i] for i in config[SOFT][GPU]]

    mem_limit = False if gpu_count == 0 else config[SOFT][GROWTH]
    tensorflow_config.set_visible_devices(gpu_devices, 'GPU')
    for device in gpu_devices:
        try:
            tensorflow_config.experimental.set_memory_growth(device, mem_limit)
        except Exception as e:
            print(f'Error tensorflow memory configuration: {e}')
