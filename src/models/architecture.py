from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


def get_cnn_model(input_shape: tuple, out_shape: int):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_shape, activation='softmax'))
    return model


def compile_model(model_architecture, input_shape: tuple, n_classes: int, loss_func, optimizer, metrics: list):
    """
    compile model architecture
    :param model_architecture: function creating model architecture
    :param input_shape: size of input data (for example (400, 400))
    :param n_classes: count of classes
    :param loss_func: function for calculate loss
    :param optimizer: model optimizer
    :param metrics: list of metrics for train
    :return: compiled model
    """
    model = model_architecture(input_shape=input_shape, out_shape=n_classes)
    model.summary()
    model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=metrics)
    return model
