#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/11/17 16:19
# @Author: Gift
# @File  : data_normalization.py 
# @IDE   : PyCharm
import numpy as np
def min_max_normalization(np_array):
    """
    最小-最大归一化 y = (x - min) / (max - min)
    :param np_array: 待归一化的数据
    :return: 归一化后的数据
    """
    min_data = np.min(np_array)
    max_data = np.max(np_array)
    return (np_array - min_data) / (max_data - min_data)
def zero_mean_normalization(np_array):
    """
    零均值归一化: y=(x - mean) / std
    :param np_array: 待归一化的数据
    :return: 归一化后的数据
    """
    return (np_array - np.mean(np_array)) / np.std(np_array)
if __name__ == '__main__':
    np.random.seed(50)
    data = np.random.randint(0, 100, size=20)
    print(data)
    print(type(data))
    print("当前数据均值：",data.mean())
    print("当前数据标准差：",data.std())
    print("当前数据最小值：",data.min())
    print("当前数据最大值：",data.max())
    min_max_data = min_max_normalization(data)
    zero_mean_data = zero_mean_normalization(data)
    print("最小-最大归一化数据：", min_max_data)
    print("零均值归一化数据：", zero_mean_data)
