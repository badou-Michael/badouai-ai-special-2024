# -*-coding:Utf-8 -*-
# Author: zl.chen
# Time:  9:22
# Name: hash_impl
# Desc:  图像hash算法实现

import numpy as np
import cv2


def mean_hash(img: np.ndarray) -> str:
    """
    均值hash
    :param img:
    :return:
    """

    return "".join(np.where(img > np.mean(img), 1, 0).flatten().astype(str))


def diff_hash(img: np.ndarray) -> str:
    """
    差值hash
    :param img:
    :return:
    """
    hash_str = ""
    for i in range(8):
        for j in range(8):
            if img[i, j] > img[i, j+1]:
                hash_str += "1"
            else:
                hash_str += "0"
    return hash_str


def hamming_distance(d1: str, d2: str) -> int:
    """
    计算两个字符串的汉明距离
    :param d1:
    :param d2:
    :return:
    """
    if len(d1) != len(d2):
        raise ValueError("d1 length not equal d2 length")
    return sum([1 if x == y else 0 for x, y in zip(d1, d2)])


if __name__ == '__main__':
    data = cv2.imread("./lenna.jpg", 0)
    data = cv2.resize(data, (8, 8), interpolation=cv2.INTER_CUBIC)
    data1 = cv2.resize(data, (9, 8), interpolation=cv2.INTER_CUBIC)
    mh = mean_hash(data)

    dh = diff_hash(data1)
    print("mean hash str: {}".format(mh))
    print("diff hash str: {}".format(dh))
    hd = hamming_distance(mh, dh)
    print("hamming distance is {}".format(hd))
