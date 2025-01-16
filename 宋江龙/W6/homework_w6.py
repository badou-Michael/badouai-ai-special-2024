#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/10/30 22:47
@Author  : Mr.Long
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

from util_tools.util_tools import UtilTools


class HomeworkW6(object):

    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def warp_perspective_matrix(self):
        assert (self.src.shape[0] == self.dst.shape[0]) and self.src.shape[0] >= 4
        nums = self.src.shape[0]
        a = np.zeros((2 * nums, 8))  # A*warpMatrix=B
        b = np.zeros((2 * nums, 1))
        for i in range(0, nums):
            a_i = self.src[i, :]
            b_i = self.dst[i, :]
            a[2 * i, :] = [a_i[0], a_i[1], 1, 0, 0, 0, -a_i[0] * b_i[0], -a_i[1] * b_i[0]]
            b[2 * i] = b_i[0]
            a[2 * i + 1, :] = [0, 0, 0, a_i[0], a_i[1], 1, -a_i[0] * b_i[1], -a_i[1] * b_i[1]]
            b[2 * i + 1] = b_i[1]
        a = np.mat(a)
        # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
        warp_matrix = a.I * b  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
        # 之后为结果的后处理
        warp_matrix = np.array(warp_matrix).T[0]
        warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
        warp_matrix = warp_matrix.reshape((3, 3))
        return warp_matrix

    def perspective_transform(self, image_array):
        """
        透视转换
        :return:
        """
        image_array_copy = image_array.copy()
        warp_matrix = self.warp_perspective_matrix()
        # warp_matrix = cv2.getPerspectiveTransform(self.src, self.dst)
        result = cv2.warpPerspective(image_array_copy, warp_matrix, (512, 512))
        return result

    def k_means(self, image_array):
        rows, cols = image_array.shape[:]
        # 图像二维像素转换为一维
        data = image_array.reshape((rows * cols, 1))
        data = np.float32(data)
        # 停止条件 (type,max_iter,epsilon)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # 设置标签
        flags = cv2.KMEANS_RANDOM_CENTERS
        # K-Means聚类 聚集成4类
        compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
        # 生成最终图像
        dst = labels.reshape((image_array.shape[0], image_array.shape[1]))
        # 用来正常显示中文标签
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 显示图像
        titles = [u'原始图像', u'聚类图像']
        images = [image_array, dst]
        for i in range(2):
            plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'), plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()



if __name__ == '__main__':
    image_path = UtilTools.get_file_path('photo1.jpg')
    image_array = cv2.imread(image_path)
    src_vertex = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst_vertex = np.float32([[0, 0], [512, 0], [0, 512], [512, 512]])
    homework_w6 = HomeworkW6(src_vertex, dst_vertex)
    result = homework_w6.perspective_transform(image_array)
    cv2.imshow("src", image_array)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    image_path_1 = UtilTools.get_file_path('lenna.png')
    image_array_1 = cv2.imread(image_path_1, 0)
    homework_w6.k_means(image_array_1)
