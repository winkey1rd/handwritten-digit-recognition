import os.path
from datetime import datetime

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from configuration.literals import *


def train_model(train_data: tuple, val_data: tuple, model, datagen, сheckpoint, reduce_lr, stopping, tensorboard,
                fit_set):
    """
    train model with loaded dataset
    :param datagen: 
    :param train_data: path to data and json
    :param val_data: path to save model
    :param model: for training
    :param сheckpoint: model checkpoints
    :param reduce_lr: class and filename separator
    :param stopping: size of batch for train model
    :param tensorboard: the maximum num elements that will be buffered when prefetching
    :param fit_set: seed for random
    """
    fit_config = (NAME, EPOCH, BATCH, VERBOSE)
    fit_values = list()
    for i, conf in enumerate(fit_config):
        fit_values.append(fit_set[conf])
    name, epochs, batch_size, verbose = fit_values

    early_stopping = EarlyStopping(monitor=stopping[MONITOR],
                                   mode=stopping[MODE],
                                   verbose=stopping[VERBOSE],
                                   patience=stopping[PATIENCE],
                                   min_delta=stopping[DELTA],
                                   baseline=stopping[BASELINE],
                                   restore_best_weights=stopping[B_WEIGHTS])

    model_info = '' if сheckpoint[BEST] else f'-{{epoch}}'
    check = ModelCheckpoint(f'models/{name}-{datetime.now().strftime("%Y%m%d")}{model_info}',
                            monitor=сheckpoint[MONITOR],
                            mode=сheckpoint[MODE],
                            verbose=сheckpoint[VERBOSE],
                            save_best_only=сheckpoint[BEST],
                            save_freq=сheckpoint[S_FREQ],
                            period=сheckpoint[PERIOD],
                            initial_value_threshold=сheckpoint[THRESHOLD],
                            save_weights_only=сheckpoint[S_WEIGHTS],
                            options=сheckpoint[OPTIONS])

    log_dir = f"models/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir,
                              update_freq=tensorboard[U_FREQ],
                              histogram_freq=tensorboard[H_FREQ],
                              embeddings_freq=tensorboard[E_FREQ],
                              write_graph=tensorboard[WRITE_G],
                              write_images=tensorboard[WRITE_I],
                              write_steps_per_second=tensorboard[STEPS],
                              profile_batch=tensorboard[P_BATCH])

    learning_rate_reduction = ReduceLROnPlateau(monitor=reduce_lr[MONITOR],
                                                mode=reduce_lr[MODE],
                                                verbose=reduce_lr[VERBOSE],
                                                patience=reduce_lr[PATIENCE],
                                                factor=reduce_lr[FACTOR],
                                                cooldown=reduce_lr[COOLDOWN],
                                                min_delta=reduce_lr[M_DELTA],
                                                min_lr=reduce_lr[MIN])

    dataset = ImageDataGenerator(rotation_range=datagen[ROTATION],
                                 zoom_range=datagen[ZOOM],
                                 width_shift_range=datagen[X_SHIFT],
                                 height_shift_range=datagen[Y_SHIFT])
    train_steps = int(train_data[0].shape[0] * datagen[MULT] / batch_size)

    model.fit(dataset.flow(*train_data, batch_size=batch_size),
              validation_data=val_data,
              epochs=epochs,
              steps_per_epoch=train_steps,
              callbacks=[check, learning_rate_reduction, early_stopping, tensorboard],
              verbose=verbose)
