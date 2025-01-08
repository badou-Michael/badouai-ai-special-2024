#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/17 21:14
@Author  : Mr.Long
"""

import cv2

from common.util_tools import UtilTools


class HomeworkW8(object):

    def __init__(self, img_array):
        self.img_array = img_array

    def average_hash(self):
        """均值hash算法
        1, 图片缩放为8*8，保留结构，除去细节
        2，灰度化：转换为灰度图。
        3, 求平均值：计算灰度图所有像素的平均值
        4, 比较：像素值大于平均值记作1，相反记作0，总共64位
        5, 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）
        """
        resize_img = cv2.resize(self.img_array, (8, 8), interpolation=cv2.INTER_CUBIC)
        gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
        sum_img, hash_str = 0, ''
        for i in range(8):
            for j in range(8):
                sum_img += gray_img[i, j]
        avg_img = sum_img / 64
        for i in range(8):
            for j in range(8):
                if gray_img[i, j] > avg_img:
                    hash_str += '1'
                else:
                    hash_str += '0'
        return hash_str

    def difference_hash(self):
        """
        1. 缩放：图片缩放为8*9，保留结构，除去细节。
        2. 灰度化：转换为灰度图。
        3. 求平均值：计算灰度图所有像素的平均值。 ---这一步没有，只是为了与均值哈希做对比
        4. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，八个差值，有8行，总共64位
        5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
        6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不相同位数越少，图片越相似。
        :return:
        """
        resize_img = cv2.resize(self.img_array, (9, 8), interpolation=cv2.INTER_CUBIC)
        gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
        hash_str = ''
        for i in range(8):
            for j in range(8):
                if gray_img[i, j] > gray_img[i, j + 1]:
                    hash_str += '1'
                else:
                    hash_str += '0'
        return hash_str





if __name__ == '__main__':
    image_path = UtilTools.get_file_path('lenna.png')
    image_array = cv2.imread(image_path)
    homework_w8 = HomeworkW8(image_array)
    print(homework_w8.average_hash())
    print(homework_w8.difference_hash())


