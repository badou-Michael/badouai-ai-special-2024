# -*- coding: utf-8 -*-
"""
@Author: zl.chen
@Name: canny_impl.py
@Time: 2024/10/31 14:01
@Desc: 透视转换实现
"""

import cv2

import numpy as np
import matplotlib.pyplot as plt


def warp_affine_impl(src: np.ndarray, dst: np.ndarray):
    """
    透视变换实现
    :param src: 图像顶点坐标矩阵
    :param dst: 纸张顶点坐标矩阵
    :return:
    """
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]

    a = np.zeros((2 * nums, 8))
    b = np.zeros((2 * nums, 1))

    # 转换矩阵 a * t_matrix = b
    for i in range(0, nums):
        a_i = src[i, :]
        b_i = dst[i, :]
        a[2 * i, :] = [a_i[0], a_i[1], 1, 0, 0, 0, -a_i[0] * b_i[0], -a_i[1] * b_i[0]]
        b[2 * i, :] = b_i[0]
        a[2 * i + 1, :] = [0, 0, 0, a_i[0], a_i[1], 1, -a_i[0] * b_i[1], -a_i[1] * b_i[1]]
        b[2 * i + 1, :] = b_i[1]

    warp_matrix = np.mat(a).I * b

    warp_matrix = np.array(warp_matrix).T[0]
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)
    warp_matrix.reshape(3, 3)
    return warp_matrix


def cv2_warp_affine():
    """
    cv2 透视转换

    :return:
    """

    img = cv2.imread("./photo1.jpg")

    result = img.copy()

    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    # 获取透视矩阵
    warp_matrix = cv2.getPerspectiveTransform(src, dst)

    rst = cv2.warpPerspective(result, warp_matrix, (337, 488))

    cv2.imshow("original", img)
    cv2.imshow("warp_affine", rst)
    cv2.waitKey(0)


def k_means_impl(src: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    """
    k—means 聚类实现
    :param src: 图像数据
    :param n_clusters 聚类的类别
    :return:
    """

    # 数据转换为一维度float

    data = np.float32(src.reshape((-1, 3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 聚集
    compactness, labels, centers = cv2.kmeans(data, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 数据转回int8
    centers = np.uint8(centers)
    result = centers[labels.flatten()].reshape(src.shape)

    # 图像转换RGB显示
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


if __name__ == '__main__':
    source = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
    destination = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])
    w_matrix = warp_affine_impl(source, destination)
    cv2_warp_affine()

    img = cv2.imread('./lenna.jpg', cv2.IMREAD_GRAYSCALE)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
              u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=64']
    data = [k_means_impl(img, i) for i in [2, 4, 6, 8, 16, 64]]
    data.insert(0, img)
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(data[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
