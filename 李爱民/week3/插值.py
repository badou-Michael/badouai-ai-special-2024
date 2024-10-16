# -*- coding: utf-8 -*-
"""
@Author: 李爱民
@Desc: 数字化图像
"""
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt


def nearest_interpolation(src_img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    最邻近值插值
    :param src_img: 原始图像信息
    :param shape: 缩扩放后图像的分辨率
    :return: 目标图像数据
    """

    src_width, src_height, channels = src_img.shape
    dst_width, dst_height = shape
    dst_img = np.zeros((dst_width, dst_height, channels), src_img.dtype)
    scale_x, scale_y = dst_width/src_width, dst_height/src_height
    for i in range(dst_width):
        for j in range(dst_height):
            x = int(i/scale_x + 0.5)
            y = int(j/scale_y + 0.5)
            dst_img[i, j] = src_img[x, y]

    return dst_img


def bilinear_interpolation(src_img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    双线性插值法
    :param src_img:
    :param shape:
    :return:
    """
    src_width, src_height, channels = src_img.shape
    dst_width, dst_height = shape
    if src_height == dst_height and src_width == dst_width:
        return src_img.copy()
    dst_img = np.zeros((dst_width, dst_height, channels), src_img.dtype)
    scale_x, scale_y = src_width/dst_width, src_height/dst_height

    for c in range(channels):
        for dst_y in range(dst_height):
            for dst_x in range(dst_width):
                # 几何中心重合公式 :src_x +0.5 = (dst_x +0.5) * scale_x 获取源图像坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 获取原坐标的四周坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_width - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_height - 1)

                # 单线性插值公式: y-y1/x-x1= y2-y1/x2-x1 --> y = ((x2-x)/(x2-x1))*y1 +((x1-x)/(x2-x1))*y2
                tmp0 = (src_x1-src_x) * src_img[src_y0, src_x0, c] + (src_x-src_x0) * src_img[src_y0, src_x1, c]
                tmp1 = (src_x1-src_x) * src_img[src_y1, src_x0, c] + (src_x-src_x0) * src_img[src_y1, src_x1, c]
                dst_img[dst_y, dst_x, c] = int((src_y1-src_y) * tmp0 + (src_y-src_y0) * tmp1)
    return dst_img


def get_histogram(src_img: np.ndarray, is_graying: bool = True, is_equal: bool = False):
    """
    获取图像的直方图
    :param src_img: 原图像数据
    :param is_graying: 是否灰度化
    :param is_equal 是否均衡化
    :return:
    """

    plt.figure()
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    if is_graying:
        title = "灰度直方图" if not is_equal else "灰度直方图均衡化"
        # 灰度化
        plt.title(title)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

        if is_equal:
            src_img = cv2.equalizeHist(src_img)

        hist = cv2.calcHist([src_img], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
    else:
        title = "彩色直方图" if not is_equal else "彩色直方图均衡化"
        plt.title(title)
        channels = cv2.split(src_img)
        if not is_equal:
            for chan, color in zip(channels, ("b", "g", "r")):
                hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
                plt.plot(hist, color=color)
                plt.xlim([0, 256])
                plt.show()

        else:

            result = [cv2.equalizeHist(channel) for channel in channels]
            data = cv2.merge((result[0], result[1], result[2]))
            cv2.imshow("data", data)


if __name__ == '__main__':
    import cv2
    img = cv2.imread("D:\lenna.png")
    nearest_result = nearest_interpolation(img, (300, 300))
    cv2.imshow("nearest interpolation", nearest_result)
    cv2.imshow("original", img)
    bilinear_result = bilinear_interpolation(img, (600, 600))
    cv2.imshow("bilinear interpolation", bilinear_result)
    get_histogram(img, False, True)
    cv2.waitKey(0)

    """
    几何中心重合证明x=1/2
    (M-1)/2 + x = ((N-1)/2 + x) * M/N 
    解:
        M/2 -1/2 + x = M/N - M/2N + x * M/N
        (1-M/N) * x = -1/2 * M/N + 1/2
        (1-M/N) * x = 1/2 (1-M/N)
        x = 1/2         
