# -*- coding: utf-8 -*-
"""
@Author: zl.chen
@Name: image_noise.py
@Time: 2024/9/23 10:26
@Desc: 图像噪声处理
"""
import random

import cv2
import numpy as np
from skimage.util import random_noise
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def deal_noise(src: np.ndarray, is_manual: bool = True, dtype: str = "gaussian", **kwargs) -> np.ndarray:
    """
    噪声处理
    :param src: 图像数据
    :param is_manual: 是否手动
    :param dtype: 噪声类型
    :param kwargs: 关键字参数
    :return: 处理后的图像数据
    """
    dst = src.copy()
    percent = kwargs.get("percent", 0.8)
    means = kwargs.get("means", 0.0)  # 高斯分布的均值
    sigma = kwargs.get("sigma", 1.0)  # 高斯分布的方差
    width, height = src.shape[:2]
    random_num = int(percent * width * height)

    if is_manual:
        for i in range(random_num):
            # 边缘不处理
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if dtype == "gaussian":
                # 高斯噪声处理
                dst[x,y] = src[x,y] + random.gauss(mu=means, sigma=sigma)
            else:
                # 椒盐处理
                if random.random() <= 0.5:
                    dst[x,y] = 0
                else:
                    dst[x,y] = 255
    else:
        # 接口调用
        if dtype == "gaussian":
            # 高斯接口
            dst = random_noise(dst, mode="gaussian", mean=means, var=sigma)
        else:
            # 椒盐接口
            dst = random_noise(dst, mode="s&p", amount=0.5)

    return dst


def deal_pca(src: np.ndarray, dim: int, is_manual: bool = True) -> np.ndarray:
    """
    主成分分析
    :param src: 原始数据
    :param dim: 降维的维度
    :param is_manual 是否手动
    :return: 降维后的数据
    """
    dst = src.copy()
    if dim >= dst.ndim:
        raise ValueError(f"dim should lt src.dim")
    if is_manual:
        # 中心化
        dst = dst - dst.mean(axis=0)
        # 求协方差
        covariance = np.dot(dst.T, dst) / dst.shape[0]
        # 求协方差的特征值和特征向量
        val, vectors = np.linalg.eig(covariance)
        # 降维矩阵：特征向量按特征值降序去取前维度个
        matrix = vectors[:, np.argsort(-val)[:dim]]
        # 降维
        dst = np.dot(dst, matrix)
    else:
        # 中心化
        dst_std = StandardScaler().fit_transform(dst)
        # 调用接口
        dst = PCA(n_components=dim).fit_transform(dst_std)
    return dst


if __name__ == '__main__':
    img = cv2.imread('./lenna.png', 0)
    cv2.imshow('lenna', img)
    manual_gaussian_img = deal_noise(img, percent=0.7, means=2.0, sigma=1.0)
    manual_sp_img = deal_noise(img, dtype="sp", percent=0.6)
    interface_gaussian_img = deal_noise(img, False, means=0.7, sigma=2.0)
    interface_sp_img = deal_noise(img, False, dtype="sp")
    cv2.imshow('manual_gaussian_img', manual_gaussian_img)
    cv2.imshow('manual_sp_img', manual_sp_img)
    cv2.imshow('interface_gaussian_img', interface_gaussian_img)
    cv2.imshow('interface_sp_img', interface_sp_img)
    cv2.waitKey(0)

    deal_pca(np.random.rand(4, 3), dim=1)
    deal_pca(np.random.rand(4, 5), dim=1, is_manual=False)
