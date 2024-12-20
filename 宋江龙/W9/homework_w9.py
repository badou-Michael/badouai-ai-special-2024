#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/19 21:17
@Author  : Mr.Long
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models




class HomeworkW9(object):

    def normalization_one(self, x):
        """
        归一化（0~1）：x_=(x−x_min)/(x_max−x_min)
        """
        return [(float(i) - min(x)) / float((max(x) - min(x))) for i in x]

    def normalization_two(self, x):
        """
        归一化（-1~1）:x_=(x-x_mean)/(x_max-x_min)
        """
        return [(float(i) - np.mean(x)) / float((max(x) - min(x))) for i in x]

    def zero_mean_normalization(self, x):
        """
        零均值归一化：(x-μ)/σ，μ是样本均值，σ是样本标准差
        """
        x_mean = np.mean(x)
        sq_sum = sum([(i - x_mean)**2 for i in x])
        std_dev = math.sqrt(sq_sum / len(x))
        return [(i - x_mean) / std_dev for i in x]

    def keras_new(self):
        """
        将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
        train_images是用于训练系统的手写数字图片;
        train_labels是用于标注图片的信息;
        test_images是用于检测系统训练效果的图片；
        test_labels是test_images图片对应的数字标签。
        """
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        network = models.Sequential()
        network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
        network.add(layers.Dense(10, activation='softmax'))
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        train_images = train_images.reshape((60000, 28 * 28))
        train_images = train_images.astype('float32') / 255
        test_images = test_images.reshape((10000, 28 * 28))
        test_images_new = test_images.astype('float32') / 255
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
        network.fit(train_images, train_labels, epochs=5, batch_size=128)
        test_loss, test_acc = network.evaluate(test_images_new, test_labels, verbose=1)
        print(test_loss)
        print('test_acc', test_acc)
        res = network.predict(test_images)
        for i in range(res[1].shape[0]):
            if res[1][i] == 1:
                print("the number for the picture is : ", i)
                break



if __name__ == '__main__':
    lst = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
           11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    hw9 = HomeworkW9()
    hw9.keras_new()
    # cs = []
    # for i in lst:
    #     c = lst.count(i)
    #     cs.append(c)
    # n = hw9.normalization_two(lst)
    # z = hw9.zero_mean_normalization(lst)
    # plt.plot(lst, cs)
    # plt.plot(z, cs)
    # plt.show()
