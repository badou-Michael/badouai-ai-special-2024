# -*- coding: utf-8 -*-
"""
@Author: zl.chen
@Name: canny_impl.py
@Time: 2024/10/16 16:25
@Desc: canny 边缘检测算法实现
"""

from typing import Tuple
import cv2
import numpy as np
from itertools import product


def gaussian_blur(image: np.ndarray, size: int = 5, sigma=1.0) -> np.ndarray:
    """
    高斯滤波
    :param image: 原始数据
    :param size:  高斯核大小
    :param sigma:  高斯方差
    :return:
    """

    # 根据高斯公式获取
    gaussian_kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)), (size, size))
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

    # 数据填充
    pad_with = size // 2
    padding = np.pad(image, ((pad_with, pad_with), (pad_with, pad_with)))

    dst = np.zeros(image.shape)
    # 卷积
    width, height = image.shape
    for w, h in product(range(width), range(height)):
        dst[w, h] = np.sum(padding[w:w + size, h:h + size] * gaussian_kernel)

    return dst


def sobel_filters(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    sobel算子求梯度
    :param image: 高斯滤波后数据
    :return: 梯度数据，梯度方向
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_x, gradient_y, gradient = np.zeros(image.shape), np.zeros(image.shape), np.zeros(image.shape)
    pad_with = sobel_x.shape[0] // 2
    padding = np.pad(image, ((pad_with, pad_with), (pad_with, pad_with)))
    width, height = image.shape
    for i, j in product(range(width), range(height)):
        gradient_x[i, j] = np.sum(padding[i:i + 3, j:j + 3] * sobel_x)
        gradient_y[i, j] = np.sum(padding[i:i + 3, j:j + 3] * sobel_y)
        gradient[i, j] = np.sqrt(gradient_x[i, j] ** 2 + gradient_y[i, j] ** 2)

    gradient_x[gradient_x == 0] = 1e-9

    direction = gradient_y / gradient_x
    return gradient, direction


def non_max_suppression(magnitude, direction):
    """
    非极大值抑制
    :param magnitude: 梯度数据
    :param direction:  梯度方向
    :return:
    """
    rows, cols = magnitude.shape
    suppressed = np.zeros((rows, cols))
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            adjacent_8 = magnitude[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if direction[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (adjacent_8[0, 1] - adjacent_8[0, 0]) / direction[i, j] + adjacent_8[0, 1]
                num_2 = (adjacent_8[2, 1] - adjacent_8[2, 2]) / direction[i, j] + adjacent_8[2, 1]
                if not (direction[i, j] > num_1 and direction[i, j] > num_2):
                    flag = False
            elif direction[i, j] >= 1:
                num_1 = (adjacent_8[0, 2] - adjacent_8[0, 1]) / direction[i, j] + adjacent_8[0, 1]
                num_2 = (adjacent_8[2, 0] - adjacent_8[2, 1]) / direction[i, j] + adjacent_8[2, 1]
                if not (direction[i, j] > num_1 and direction[i, j] > num_2):
                    flag = False
            elif direction[i, j] > 0:
                num_1 = (adjacent_8[0, 2] - adjacent_8[1, 2]) * direction[i, j] + adjacent_8[1, 2]
                num_2 = (adjacent_8[2, 0] - adjacent_8[1, 0]) * direction[i, j] + adjacent_8[1, 0]
                if not (direction[i, j] > num_1 and direction[i, j] > num_2):
                    flag = False
            elif direction[i, j] < 0:
                num_1 = (adjacent_8[1, 0] - adjacent_8[0, 0]) * direction[i, j] + adjacent_8[1, 0]
                num_2 = (adjacent_8[1, 2] - adjacent_8[2, 2]) * direction[i, j] + adjacent_8[1, 2]
                if not (direction[i, j] > num_1 and direction[i, j] > num_2):
                    flag = False

            if flag:
                suppressed[i, j] = direction[i, j]

    return suppressed


def hysteresis_thresholding(image, low_threshold, high_threshold):
    """
    双阈值检测
    :param image: 图像数据
    :param low_threshold: 低阈值
    :param high_threshold: 高阈值
    :return:
    """
    rows, cols = image.shape
    edge_index = []
    for x, y in product(range(1, rows - 1), range(1, cols - 1)):
        if image[x, y] >= high_threshold:
            image[x, y] = 255
            edge_index.append((x, y))
        elif image[x, y] < low_threshold:
            image[x, y] = 0
    while not edge_index:
        x, y = edge_index.pop()
        # 取临近8值
        adjacent_8 = image[x - 1:x + 2, y - 1:y + 2]

        if low_threshold < adjacent_8[0, 0] < low_threshold:
            image[x - 1, y - 1] = 255
            edge_index.append((x - 1, y - 1))

        if low_threshold < adjacent_8[0, 1] < low_threshold:
            image[x - 1, y] = 255
            edge_index.append((x - 1, y))

        if low_threshold < adjacent_8[0, 2] < low_threshold:
            image[x - 1, y + 1] = 255
            edge_index.append((x - 1, y + 1))

        if low_threshold < adjacent_8[1, 0] < low_threshold:
            image[x, y - 1] = 255
            edge_index.append((x, y - 1))

        if low_threshold < adjacent_8[1, 2] < low_threshold:
            image[x, y + 1] = 255
            edge_index.append((x, y + 1))

        if low_threshold < adjacent_8[2, 0] < low_threshold:
            image[x + 1, y - 1] = 255
            edge_index.append((x + 1, y - 1))

        if low_threshold < adjacent_8[2, 1] < low_threshold:
            image[x + 1, y] = 255
            edge_index.append((x + 1, y))

        if low_threshold < adjacent_8[2, 2] < low_threshold:
            image[x + 1, y + 1] = 255
            edge_index.append((x + 1, y + 1))

    for x, y in product(range(rows), range(cols)):
        if image[x, y] not in [0, 255]:
            image[x, y] = 0

    return image


def canny_edge_detection(src: np.ndarray, size: int = 5, sigma: float = 1.5) -> np.ndarray:
    """
    canny 边缘检测
    :param src: 原始数据
    :param size: 高斯核大小
    :param sigma: 高斯核方差
    :return:
    """

    # 高斯滤波
    gauss = gaussian_blur(src, size, sigma)
    # sobel算子求梯度
    edge, angle = sobel_filters(gauss)

    # 非极大值抑制
    suppressed = non_max_suppression(edge, angle)

    # 双阈值检测
    low = suppressed.mean() * 0.5
    high = low * 3

    edges = hysteresis_thresholding(suppressed, low, high)

    return edges


if __name__ == '__main__':
    img = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
    result = canny_edge_detection(img)
    cv2.imshow('canny', result)
    cv2.waitKey(0)
