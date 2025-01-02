#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：model.py
@IDE     ：PyCharm 
@Author  ：chung rae
@Date    ：2025/1/2 21:27 
@Desc : 
"""
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
import numpy as np
from tensorflow_core import Tensor

from config import NORM_EPSILON, NORM_DECAY, ANCHORS_PATH, CLASSES_PATH

class Yolo:
    def __init__(
            self,
            norm_eps: float=NORM_EPSILON,
            norm_decay: float=NORM_DECAY,
            anchors_path: Path=ANCHORS_PATH,
            classes_path: Path=CLASSES_PATH,
            pre_trained: bool=True,
    ):
        self.norm_eps = norm_eps
        self.norm_decay = norm_decay
        self.pre_trained = pre_trained
        self.anchors = self.__get_anchors(anchors_path)
        self.classes = self.__get_classes(classes_path)

    @staticmethod
    def __get_anchors(anchors_path: Path) -> np.ndarray:
        with open(anchors_path, 'r') as f:
            anchors = f.readline()
        anchors = list(map(lambda x: float(x), anchors.split(',')))
        anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    @staticmethod
    def __get_classes(classes_path: Path) -> List[str]:
        with open(classes_path, 'r') as f:
            classes = f.readlines()
        classes = list(map(lambda x: x.strip(), classes))
        return classes

    @staticmethod
    def _conv2d_layer(inputs: Tensor, filters: int, kernel_size: int, name: str, use_bias: bool=False, strides: int = 1) -> Tensor:
        conv = tf.layers.conv2d(
            inputs=inputs, filters=filters,
            kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
            padding=('SAME' if strides == 1 else 'VALID'),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name)
        return conv


    @staticmethod
    def _batch_normalization_layer(input_layer: Tensor, name: str=None, training: bool=True, normal_decay: float=NORM_DECAY, normal_eps: float=NORM_EPSILON) -> Tensor:
        """

        :param input_layer:
        :param name:
        :param training:
        :param normal_decay:
        :param normal_eps:
        :return:
        """
        bn_layer = tf.layers.batch_normalization(inputs=input_layer,
                                                 momentum=normal_decay, epsilon=normal_eps, center=True,
                                                 scale=True, training=training, name=name)
        return tf.nn.leaky_relu(bn_layer, alpha=0.1)


    def _residual_block(self, inputs: Tensor, filters: int, block_num: int, conv_idx: int, training: bool=True, normal_decay: float=NORM_DECAY, normal_eps: float=NORM_EPSILON) -> Tuple[Tensor, int]:
        """

        :param inputs:
        :param filters:
        :param block_num:
        :param conv_idx:
        :param training:
        :param normal_decay:
        :param normal_eps:
        :return:
        """
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self._conv2d_layer(inputs, filters, kernel_size=3, strides=2, name="conv2d_" + str(conv_idx))
        layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_idx), training=training,
                                                normal_decay=normal_decay, normal_eps=normal_eps)
        conv_idx += 1
        for _ in range(block_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters // 2, kernel_size=1, strides=1, name="conv2d_" + str(conv_idx))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_idx), training=training, normal_decay=normal_decay, normal_eps=normal_eps)
            conv_idx += 1
            layer = self._conv2d_layer(layer, filters, kernel_size=3, strides=1, name="conv2d_" + str(conv_idx))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_idx),
                                                    training=training, normal_decay=normal_decay, normal_eps=normal_eps)
            conv_idx += 1
            layer += shortcut
        return layer, conv_idx

    def _darknet53(self, inputs: Tensor, conv_idx: int, training: bool=True, normal_decay: float=NORM_DECAY, normal_eps: float=NORM_EPSILON) -> Tuple[Tensor,Tensor,Tensor, int]:
        """

        :param inputs:
        :param conv_idx:
        :param training:
        :param normal_decay:
        :param normal_eps:
        :return:
        """
        with tf.variable_scope('darknet53'):
            conv = self._conv2d_layer(inputs, filters=32, kernel_size=3, strides=1,
                                      name="conv2d_" + str(conv_idx))
            conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_idx),
                                                   training=training, normal_decay=normal_decay, normal_eps=normal_eps)
            conv_idx += 1
            conv, conv_idx = self._residual_block(conv, conv_idx=conv_idx, filters=64, block_num=1, training=training, normal_decay=normal_decay, normal_eps=normal_eps)
            conv, conv_idx = self._residual_block(conv, conv_idx=conv_idx, filters=128, block_num=2, training=training, normal_decay=normal_decay, normal_eps=normal_eps)
            conv, conv_idx = self._residual_block(conv, conv_idx=conv_idx, filters=256, block_num=8, training=training, normal_decay=normal_decay, normal_eps=normal_eps)
            route1 = conv

            conv, conv_idx = self._residual_block(conv, conv_idx=conv_idx, filters=512, block_num=8, training=training, normal_decay=normal_decay, normal_eps=normal_eps)
            route2 = conv
            conv, conv_idx = self._residual_block(conv, conv_idx=conv_idx, filters=1024, block_num=4, training=training, normal_decay=normal_decay, normal_eps=normal_eps)
            return route1, route2, conv, conv_idx

    def _yolo_block(self, inputs: Tensor, filters: int, outputs_filter: int, conv_idx: int, training: bool=True, normal_decay: float=NORM_DECAY, normal_eps: float=NORM_EPSILON) -> Tuple[Tensor, Tensor, int]:
        """

        :param inputs:
        :param filters:
        :param outputs_filter:
        :param conv_idx:
        :param training:
        :param normal_decay:
        :param normal_eps:
        :return:
        """

        conv = self._conv2d_layer(inputs, filters=filters, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_idx))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_idx), training=training,
                                               normal_decay=normal_decay, normal_eps=normal_eps)
        conv_idx += 1
        conv = self._conv2d_layer(conv, filters=filters * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_idx))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_idx), training=training,
                                               normal_decay=normal_decay, normal_eps=normal_eps)
        conv_idx += 1
        conv = self._conv2d_layer(conv, filters=filters, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_idx))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_idx), training=training,
                                               normal_decay=normal_decay, normal_eps=normal_eps)
        conv_idx += 1
        conv = self._conv2d_layer(conv, filters=filters * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_idx))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_idx), training=training,
                                               normal_decay=normal_decay, normal_eps=normal_eps)
        conv_idx += 1
        conv = self._conv2d_layer(conv, filters=filters, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_idx))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_idx), training=training,
                                               normal_decay=normal_decay, normal_eps=normal_eps)
        conv_idx += 1
        route = conv
        conv = self._conv2d_layer(conv, filters=filters * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_idx))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_idx), training=training,
                                               normal_decay=normal_decay, normal_eps=normal_eps)
        conv_idx += 1
        conv = self._conv2d_layer(conv, filters=outputs_filter, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_idx), use_bias=True)
        conv_idx += 1
        return route, conv, conv_idx

    def infer(self, inputs: Tensor, num_anchors: int, num_classes: int, training: bool=True) -> List[Tensor]:
        conv_idx = 1
        conv2d_26, conv2d_43, conv, conv_idx = self._darknet53(
            inputs, conv_idx, training=training, normal_decay=self.norm_decay,
            normal_eps=self.norm_eps,
        )

        with tf.variable_scope('yolo'):
            conv2d_57, conv2d_59, conv_idx = self._yolo_block(conv, 512, num_anchors *(num_classes + 5), conv_idx, training=training, normal_decay=self.norm_decay, normal_eps=self.norm_eps)
            conv2d_60 = self._conv2d_layer(conv2d_57, filters=256, kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_idx))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name="batch_normalization_" + str(conv_idx),
                                                        training=training,normal_decay=self.norm_decay, normal_eps=self.norm_eps)
            conv_idx += 1
            unsample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]], name='unsample_0')
            route0 = tf.concat([unsample_0, conv2d_43], axis=-1, name='route_0')
            conv2d_65, conv2d_67, conv_idx = self._yolo_block(route0, 256, num_anchors * (num_classes + 5),
                                                                conv_idx, training=training,
                                                                normal_decay=self.norm_decay, normal_eps=self.norm_eps)

            conv2d_68 = self._conv2d_layer(conv2d_65, filters=128, kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_idx))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name="batch_normalization_" + str(conv_idx),
                                                        training=training, normal_decay=self.norm_decay, normal_eps=self.norm_eps)
            conv_idx += 1

            unsample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='unsample_1')
            route1 = tf.concat([unsample_1, conv2d_26], axis=-1, name='route_1')

            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_idx,
                                               training=training, normal_decay=self.norm_decay, normal_eps=self.norm_eps
                                               )

            return [conv2d_59, conv2d_67, conv2d_75]