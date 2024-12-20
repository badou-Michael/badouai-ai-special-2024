from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    model = Sequential()
    # layer 1
    model.add(Conv2D(
        filters=96,
        kernel_size=(11, 11),
        strides=(4, 4),
        padding='valid',
        activation='relu',
        input_shape=input_shape
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='valid'))
    # layer 2
    model.add(Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding='valid',
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='valid'))
    # layer 3
    model.add(Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation='relu'
    ))
    # layer 4
    model.add(Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation='relu'
    ))
    # layer 5
    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='valid'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation='softmax'))

    return model
