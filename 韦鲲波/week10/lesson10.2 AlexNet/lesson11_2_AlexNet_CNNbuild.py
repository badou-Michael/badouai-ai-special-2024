from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam
import os

class AlexNet:
    # 初始化模型
    network = Sequential()
    def __init__(self, input_shape=(227, 227, 3), output_shape=2):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.conv1()
        self.conv2()
        self.conv3()
        self.conv4()
        self.model = self.fc()

    def conv1(self, filters=48):
        # 先添加一个卷积层
        self.network.add(
            Conv2D(
                filters=filters,
                kernel_size=(11, 11),
                strides=(4, 4),
                input_shape=self.input_shape,
                activation='relu',
            )
        )

        self.network.add(BatchNormalization())

        # 定义一个池化层
        self.network.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
            )
        )

    def conv2(self, filters=128):
        # 定义一个卷积层
        self.network.add(
            Conv2D(
                filters=filters,
                kernel_size=(5, 5),
                padding='same',
                activation='relu',
            )
        )

        # 定义一个池化层
        self.network.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
            )
        )

    def conv3(self, filters=192):
        # 定义一个卷积层
        self.network.add(
            Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            )
        )

        # 定义一个卷积层
        self.network.add(
            Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                padding='same',
            )
        )

    def conv4(self, filters=128):
        # 定义一个卷积层
        self.network.add(
            Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            )
        )

        # 定义一个池化层
        self.network.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
            )
        )

        # 最后一层，给做成一维矩阵
        self.network.add(Flatten())

    def fc(self):
        # self.network.add(
        #     Dense(
        #         units=4608,
        #         activation='relu',
        #     )
        # )
        #
        # self.network.add(Dropout(0.25))

        self.network.add(
            Dense(
                units=1024,
                activation='relu',
            )
        )

        self.network.add(Dropout(0.25))

        self.network.add(
            Dense(
                units=1024,
                activation='relu',
            )
        )

        self.network.add(Dropout(0.25))

        self.network.add(
            Dense(
                units=self.output_shape,
                activation='softmax',
            )
        )

        return self.network
