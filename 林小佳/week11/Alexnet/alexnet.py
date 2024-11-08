from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam


def AlexNet(input_shape=(224, 224, 3), output_shape=2):  # 图像类别为两类
    model = Sequential()  # 创建模型的实例化对象model

    # 第一卷积模块
    # 卷积层，输出结果为48层
    model.add(Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), padding='valid', input_shape=input_shape,
                     activation='relu'))

    model.add(BatchNormalization())  # 批归一化

    # 池化层
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 第2卷积模块
    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    model.add(Flatten())    # 重塑为一维

    # 全连接层FC1
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    # 全连接层FC2
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation='softmax'))

    return model
