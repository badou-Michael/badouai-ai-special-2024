#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：chung rae
@Date    ：2025/1/2 19:26 
@Desc : 
"""
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np
import tensorflow as tf


def load_weights(variables: List, weight_fp: Path) -> List:
    """

    :param variables:
    :param weight_fp:
    :return:
    """

    with open(weight_fp, "rb") as f:
        _ = np.fromfile(f, np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

    ptr, i, assign_ops = 0, 0, []
    while i < len(variables) - 1:
        var1, var2 = variables[i], variables[i + 1]
        if "conv2d" in var1.name.split("/")[-2]:
            if "batch_normalization" in var2.name.split("/")[-2]:
                gama, beta, mean, var = variables[i + 1: i + 5]
                batch_normal_vars = [beta, gama, mean, var]
                for var in batch_normal_vars:
                    shape = var.shape.as_list()
                    number_params = np.prod(shape)
                    var_weights = weights[ptr: ptr + number_params].reshape(shape)
                    ptr += number_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                i += 4
            elif "conv2d" in var2.name.split("/")[-2]:
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_number_params = np.prod(bias_shape)
                bias_weights = weights[ptr: ptr + bias_number_params].reshape(bias_shape)
                ptr += bias_number_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                i += 1

            shape = var1.shape.as_list()
            number_params = np.prod(shape)
            var_weights = weights[ptr: ptr + number_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += number_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops



def resize_image(image:Image, size: Tuple[int, int]) -> Image:
    """
    :param image:
    :param size:
    :return:
    """
    original_width, original_height = image.size
    new_width, new_height = size
    resized_img = image.resize((new_width, new_height), Image.BICUBIC)
    boxed_img = Image.new("RGB", size, (128, 128, 128))
    boxed_img.paste(resized_img, ((original_width -new_width) / 2, (original_height -new_height) / 2))
    return boxed_img
