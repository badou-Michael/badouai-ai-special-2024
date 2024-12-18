#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：convolution_neural_network.py
@IDE     ：PyCharm 
@Author  ：chung rae
@Date    ：2024/12/18 20:06 
@Desc :  resnet50 inception mobilenet implementation
"""
from pathlib import Path
from typing import List, Tuple
from keras.models import Model
from keras import Input
from keras import backend as K
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, AveragePooling2D, \
    Flatten, GlobalAveragePooling2D, DepthwiseConv2D, Reshape, Dropout
import numpy as np
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras_applications.densenet import layers
from tensorflow_core import Tensor


class BaseCNN:
    relu_max_val: int = None
    name: str = None

    def __init__(self, input_shape: List[int], weights_path: Path, img_path: Path):
        if self.name is None:
            raise ValueError('name must be set')

        self.img_path = img_path
        self.input_shape = input_shape
        self.model = None

        self.model = self.build()

        if self.model is not None:
            self.model.load_weights(weights_path.as_posix())

    def build(self):
        raise NotImplementedError

    def predict(self):
        img = load_img(self.img_path, target_size=tuple(self.input_shape[:-1]))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        pred = self.model.predict(img)
        print(decode_predictions(pred, 0))

    @classmethod
    def __conv2bnp(cls, inputs: Tensor, filters, kernel_size, strides: Tuple[int, int] = (1, 1), padding: str = "same",
                  use_bias: bool = False, conv_type: str = "max", name: str = None, relu:str = None,
                  pool_type: str = None, pool_size: Tuple[int, int] = None, pool_strides: Tuple[int, int] = None
                  ) -> Tensor:
        """
        :param inputs:
        :param filters:
        :param kernel_size:
        :param strides:
        :param padding:
        :param use_bias:
        :param name:
        :param relu
        :param pool_type:
        :param pool_size:
        :param pool_strides:
        :return:
        """
        if name is not None:
            bn_name = f"{name}_bn"
            cn_name = f"{name}_cn"
        else:
            bn_name = None
            cn_name = None
        data = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                      name=cn_name)(inputs)
        data = BatchNormalization(name=bn_name, scale=False)(data)
        if relu is not None:
            relu = lambda x: K.relu(data, max_value=cls.relu_max_val)
            data = Activation(relu)(data)


        if pool_type is not None:
            if not all([pool_size, pool_strides]):
                raise ValueError("pool_size and pool_strides must not be None")

        if pool_type == "max":
            pool = MaxPooling2D
        else:
            pool = AveragePooling2D

        if pool is not None:
            data = pool(pool_size, pool_strides)(data)
        return data


class Resnet50(BaseCNN):
    def build(self):
        img = Input(shape=self.input_shape)
        data = ZeroPadding2D(padding=(3, 3))(img)
        x = self.__conv2bnp(data, 64, (7, 7), name="conv1", relu="relu", pool_size=(3, 3), pool_strides=(2, 2))
        x = self.batch_block(x)
        x = AveragePooling2D((7, 7))(x)
        x = Flatten()(x)
        x = Dense(1000, activation="softmax")(x)
        model = Model(img, x,name="resnet50")
        return model


    def batch_block(self, data: Tensor, start: int = 2, end: int= 6) -> Tensor:
        filters = [32, 32, 128]
        for i in range(start, end):
            filters = list(map(lambda x: x * 2, filters))
            if i == 3:
                blocks =["a", "b", "c", "d"]
            elif i == 4:
                blocks = ["a", "b", "c", "d", "e", "f"]
            else:
                blocks =  ["a", "b", "c"]

            for block in blocks:
                if block == "a":
                    data = self.__block(data, 3, filters, stage=i, block=block)
                else:
                    data = self.__block(data, 3, filters, stage=i, block=block, is_identify=True)
        return data

    def __block(self, data: Tensor, kernel_size: int, filters: List[int], stage: int, block: str,
                   strides: Tuple[int, int] = (2, 2), is_identify: bool=False) -> Tensor:
        """

        :param data:
        :param kernel_size:
        :param filters:
        :param stage:
        :param block:
        :param strides:
        :param is_identify:
        :return:
        """
        conv_name, bn_name = f"res{stage}_{block}_branch", f"bn{stage}{block}_branch"
        f1, f2, f3 = filters

        x = self.__conv2bnp(data, filters=f1, kernel_size=(1, 1), strides=strides, name=f"{conv_name}2a", relu="relu")
        x = self.__conv2bnp(x, filters=f2, kernel_size=kernel_size, strides=strides, name=f"{conv_name}2b", relu="relu")
        x = self.__conv2bnp(x, filters=f3, kernel_size=(1, 1), name=f"{conv_name}2c")
        if not is_identify:
            shortcut = self.__conv2bnp(data, filters=f3, kernel_size=(1, 1), strides=strides, name=f"{conv_name}1")
        else:
            shortcut= data

        x = layers.add([x, shortcut])
        x = Activation("relu")(x)
        return x


class InceptionV3(BaseCNN):

    name = "InceptionV3"

    def build(self):
        data = Input(shape=self.input_shape)
        x = self.__conv2bnp(data, filters=32, kernel_size=(3, 3), name="conv1", padding="valid", relu="relu")
        x = self.__conv2bnp(x, filters=32, kernel_size=(3, 3), name="conv2", padding="valid", relu="relu")
        x = self.__conv2bnp(x, filters=64, kernel_size=(3, 3), name="conv3", relu="relu")
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.__conv2bnp(x, filters=80, kernel_size=(1, 1), name="conv4", relu="relu", padding="valid")
        x = self.__conv2bnp(x, filters=192, kernel_size=(1, 1), name="conv5", relu="relu", padding="valid")
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # 35*35*192 --> 35*35*256
        block1_1_1 = self.__conv2bnp(x, filters=64, kernel_size=(1, 1), name="conv6", relu="relu")
        block1_5_5 = self.__conv2bnp(x, filters=48, kernel_size=(1, 1), name="conv7", relu="relu")
        block1_5_5 = self.__conv2bnp(block1_5_5, filters=64, kernel_size=(5, 5), name="conv8", relu="relu")
        block1_3_3 = self.__conv2bnp(x, filters=64, kernel_size=(1, 1), name="conv6", relu="relu")
        block1_3_3 = self.__conv2bnp(block1_3_3, filters=96, kernel_size=(3, 3), name="conv7", relu="relu")
        block1_3_3 = self.__conv2bnp(block1_3_3, filters=96, kernel_size=(3, 3), name="conv8", relu="relu")
        block1_pool = AveragePooling2D((3, 3), strides=(2, 2), padding="same")(x)
        block1_pool = self.__conv2bnp(block1_pool, filters=32, kernel_size=(1, 1), name="conv9", relu="relu")
        x = layers.concatenate([block1_1_1, block1_5_5, block1_3_3, block1_pool], axis=3)

        # 35*35*288 --> 17*17*768
        block2_3_3 = self.__conv2bnp(x, filters=384, kernel_size=(3, 3), strides=(2,2), name="conv10", relu="relu", padding="valid")
        block2_1_1 = self.__conv2bnp(x, filters=64, kernel_size=(1, 1), name="conv11", relu="relu")
        block2_1_1= self.__conv2bnp(block2_1_1, filters=96, kernel_size=(3, 3), name="conv12", relu="relu")
        block2_1_1 = self.__conv2bnp(block2_1_1, filters=96, kernel_size=(3, 3), strides=(2, 2), name="conv13", relu="relu", padding="valid")
        block2_pool = AveragePooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.concatenate([block2_3_3, block2_1_1, block2_pool], axis=3)

        # 17*17*768 --> 17*17*768
        block3_1_1 = self.__conv2bnp(x, filters=192, kernel_size=(1, 1), name="conv14", relu="relu")
        block3_7_7 = self.__conv2bnp(x, filters=128, kernel_size=(1, 1), name="conv15", relu="relu")
        block3_7_7 = self.__conv2bnp(block3_7_7, filters=128, kernel_size=(1, 7), name="conv16", relu="relu")
        block3_7_7 = self.__conv2bnp(block3_7_7, filters=192, kernel_size=(7, 1), name="conv16", relu="relu")
        block3_7_b = self.__conv2bnp(x, filters=128, kernel_size=(1, 1), name="conv17", relu="relu")
        block3_7_b = self.__conv2bnp(block3_7_b, filters=128, kernel_size=(7, 1), name="conv18", relu="relu")
        block3_7_b = self.__conv2bnp(block3_7_b, filters=128, kernel_size=(1, 7), name="conv19", relu="relu")
        block3_7_b = self.__conv2bnp(block3_7_b, filters=128, kernel_size=(7, 1), name="conv20", relu="relu")
        block3_7_b = self.__conv2bnp(block3_7_b, filters=192, kernel_size=(1, 7), name="conv20", relu="relu")
        block3_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        block3_pool = self.__conv2bnp(block3_pool, filters=192, kernel_size=(1, 1), name="conv21", relu="relu")
        x = layers.concatenate([block3_1_1, block3_7_7, block3_7_b, block3_pool], axis=3)

        for i in range(2):
            block4_1_1 = self.__conv2bnp(x, filters=192, kernel_size=(1, 1), name="conv22", relu="relu")
            block4_7_7 = self.__conv2bnp(x, filters=160, kernel_size=(1, 1), name="conv23", relu="relu")
            block4_7_7 = self.__conv2bnp(block4_7_7, filters=160, kernel_size=(1, 7), name="conv24", relu="relu")
            block4_7_7 = self.__conv2bnp(block4_7_7, filters=192, kernel_size=(7, 1), name="conv25", relu="relu")
            block4_7_b = self.__conv2bnp(x, filters=160, kernel_size=(1, 1), name="conv26", relu="relu")
            block4_7_b = self.__conv2bnp(block4_7_b, filters=160, kernel_size=(7, 1), name="conv27", relu="relu")
            block4_7_b = self.__conv2bnp(block4_7_b, filters=160, kernel_size=(1, 7), name="conv28", relu="relu")
            block4_7_b = self.__conv2bnp(block4_7_b, filters=160, kernel_size=(7, 1), name="conv29", relu="relu")
            block4_7_b = self.__conv2bnp(block4_7_b, filters=192, kernel_size=(1,7), name="conv30", relu="relu")
            block4_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
            block4_pool = self.__conv2bnp(block4_pool, filters=192, kernel_size=(1, 1), name="conv31", relu="relu")
            x = layers.concatenate([block4_1_1, block4_7_7, block4_7_b, block4_pool], axis=3)

        block5_1_1 = self.__conv2bnp(x, filters=192, kernel_size=(1, 1), name="conv32", relu="relu")
        block5_7_7 = self.__conv2bnp(x, filters=192, kernel_size=(1, 1), name="conv33", relu="relu")
        block5_7_7 = self.__conv2bnp(block5_7_7, filters=192, kernel_size=(7, 1), name="conv34", relu="relu")
        block5_7_7 = self.__conv2bnp(block5_7_7, filters=192, kernel_size=(1, 7), name="conv35", relu="relu")
        block5_7_7_b = self.__conv2bnp(x, filters=192, kernel_size=(1, 1), name="conv36", relu="relu")
        block5_7_7_b = self.__conv2bnp(block5_7_7_b, filters=192, kernel_size=(7, 1), name="conv37", relu="relu")
        block5_7_7_b = self.__conv2bnp(block5_7_7_b, filters=192, kernel_size=(1, 7), name="conv38", relu="relu")
        block5_7_7_b = self.__conv2bnp(block5_7_7_b, filters=192, kernel_size=(7, 1), name="conv39", relu="relu")
        block5_7_7_b = self.__conv2bnp(block5_7_7_b, filters=192, kernel_size=(1, 7), name="conv40", relu="relu")
        block5_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        block5_pool = self.__conv2bnp(block5_pool, filters=192, kernel_size=(1, 1), name="conv41", relu="relu")
        x = layers.concatenate([block5_1_1, block5_7_7, block5_7_7_b, block5_pool], axis=3)


        block6_1_1 = self.__conv2bnp(x, filters=192, kernel_size=(1, 1), name="conv42", relu="relu")
        block6_1_1 = self.__conv2bnp(block6_1_1, filters=320, kernel_size=(3, 3), strides=(2, 2), name="conv43", relu="relu", padding="valid")
        block6_7_7 = self.__conv2bnp(x, filters=192, kernel_size=(1, 1), name="conv44", relu="relu")
        block6_7_7 = self.__conv2bnp(block6_7_7, filters=192, kernel_size=(1, 7), name="conv45", relu="relu")
        block6_7_7 = self.__conv2bnp(block6_7_7, filters=192, kernel_size=(7, 1), name="conv46", relu="relu")
        block6_7_7 = self.__conv2bnp(block6_7_7, filters=192, kernel_size=(3, 3), strides=(2,2), name="conv47", relu="relu", padding="valid")
        block6_pool = AveragePooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate([block6_1_1, block6_7_7, block6_pool], axis=3)

        for i in range(2):
            block7_1_1 = self.__conv2bnp(x, filters=320, kernel_size=(1, 1), name="conv48", relu="relu")
            block7_3_3 = self.__conv2bnp(x, filters=384, kernel_size=(1, 1), name="conv49", relu="relu")
            block7_3_3_1 = self.__conv2bnp(block7_3_3, filters=384, kernel_size=(1, 3), name="conv50", relu="relu")
            block7_3_3_2 = self.__conv2bnp(block7_3_3, filters=384, kernel_size=(3, 1), name="conv51", relu="relu")
            block7_3_3 = layers.concatenate([block7_3_3_1, block7_3_3_2], axis=3)
            block7_3_3_b = self.__conv2bnp(x, filters=448, kernel_size=(1, 1), name="conv52", relu="relu")
            block7_3_3_b = self.__conv2bnp(block7_3_3_b, filters=384, kernel_size=(3, 3), name="conv53", relu="relu")
            block7_3_3_b_1 = self.__conv2bnp(block7_3_3_b, filters=384, kernel_size=(1, 3), name="conv54", relu="relu")
            block7_3_3_b_2 = self.__conv2bnp(block7_3_3_b, filters=384, kernel_size=(3, 1), name="conv55", relu="relu")
            block7_3_3_b = layers.concatenate([block7_3_3_b_1, block7_3_3_b_2], axis=3)
            block7_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
            block7_pool = self.__conv2bnp(block7_pool, filters=192, kernel_size=(1, 1), name="conv56", relu="relu")
            x = layers.concatenate([block7_1_1, block7_3_3, block7_3_3_b, block7_pool], axis=3)

        x = GlobalAveragePooling2D()(x)
        x = Dense(1000, activation="softmax")(x)
        model = Model(data, x, name="inceptionV3")
        return model


class MobileNet(BaseCNN):
    name = "MobileNet"
    relu_max_val = 6

    def __depthwise_conv2d(self, inputs: Tensor, filters, depth_multiplier: int = 1, strides: Tuple[int, int]=(1, 1), block: int = 1, padding: str="same", use_bias: bool=False) -> Tensor:
        """

        :param inputs:
        :param filters:
        :param depth_multiplier:
        :param strides:
        :param block:
        :param padding:
        :param use_bias:
        :return:
        """
        x = DepthwiseConv2D(
            kernel_size=(3, 3),
            padding=padding,
            depth_multiplier=depth_multiplier,
            strides=strides,
            use_bias=use_bias,
            name=f"conv_dw_{block}",
        )(inputs)
        x = BatchNormalization(name=f"bn_dw_bn{block}")(x)
        x = Activation("relu", name=f"bn_dw_relu{block}")(x)
        x = Conv2D(filters, (1,1), strides=(1,1), use_bias=use_bias, name=f"conv_pw_conv{block}")(x)
        x = BatchNormalization(name=f"conv_pw_bn{block}")(x)
        x = Activation("relu", name=f"conv_pw_relu{block}")(x)
        return x

    def build(self):
        data = Input(shape=self.input_shape)
        depth_multiplier, dropout = 1, 1e-3
        x = self.__conv2bnp(data, filters=32, kernel_size=(3, 3), name="conv1", relu="relu")
        x = self.__depthwise_conv2d(x, 64, depth_multiplier=depth_multiplier, block=1)
        x = self.__depthwise_conv2d(x, 128, depth_multiplier=depth_multiplier, block=2, strides=(2, 2))
        x = self.__depthwise_conv2d(x, 128, depth_multiplier=depth_multiplier, block=3)
        x = self.__depthwise_conv2d(x, 256, depth_multiplier=depth_multiplier, block=4, strides=(2, 2))
        x = self.__depthwise_conv2d(x, 256, depth_multiplier=depth_multiplier, block=5)
        x = self.__depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, block=6, strides=(2, 2))
        x = self.__depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, block=7)
        x = self.__depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, block=8)
        x = self.__depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, block=9)
        x = self.__depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, block=10)
        x = self.__depthwise_conv2d(x, 512, depth_multiplier=depth_multiplier, block=11)
        x = self.__depthwise_conv2d(x, 1024, depth_multiplier=depth_multiplier, block=12, strides=(2, 2))
        x = self.__depthwise_conv2d(x, 1024, depth_multiplier=depth_multiplier, block=13)
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 1024))(x)
        x = Dropout(dropout)(x)
        x = Conv2D(1000, (1,1), padding="same")(x)
        x = Activation("softmax")(x)
        x = Reshape((1000, ))(x)
        model = Model(data, x, name="mobilenet")
        return model


if __name__ == '__main__':
    img_path = Path("./data/elephant.jpg")
    Resnet50(
        [224, 224, 3],
        Path("./data/resnet50_weights_tf_dim_ordering_tf_kernels.h5"), img_path
    ).predict()

    InceptionV3(
        [299, 299, 3],
        Path("./data/inception_v3_weights_tf_dim_ordering_tf_kernels.h5"), img_path
    ).predict()

    MobileNet(
        [224, 224, 3],
        Path("./data/mobilenet_1_0_224_tf.h5"),
        img_path
    ).predict()

