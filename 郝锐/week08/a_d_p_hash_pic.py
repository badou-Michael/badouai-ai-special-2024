#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/11/12 14:04
# @Author: Gift
# @File  : a_d_p_hash_pic.py
# @IDE   : PyCharm
#均值哈希
import cv2
import numpy as np
def cal_average_hash(img,hash_size=8):
    # 将图片转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 缩放图片到8x8
    gray = cv2.resize(gray, (hash_size, hash_size))
    # 计算均值
    avg = gray.mean()
    # 计算哈希值
    a_hash = ''
    for i in range(hash_size):
        for j in range(hash_size):
            if gray[i, j] > avg:
                a_hash += '1'
            else:
                a_hash += '0'
    return a_hash
#差值哈希
def cal_diff_hash(img,hash_size=8):
    # 将图片转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 缩放图片到8x9
    gray = cv2.resize(gray, (hash_size, hash_size+1))
    print(gray.shape)
    # 计算差值哈希值
    d_hash = ''
    for i in range(hash_size):
        for j in range(hash_size):
            if gray[i, j] > gray[i + 1, j]:
                d_hash += '1'
            else:
                d_hash += '0'
    return d_hash
# 感知哈希
def cal_p_hash(img,hash_size=32):
    #读取图像并转换为32*32的灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (hash_size, hash_size))
    # 计算DCT变换
    dct = cv2.dct(np.float32(gray))
    # 取左上角的8*8
    dct = dct[:8, :8]
    # 计算均值
    avg = dct.mean()
    # 计算哈希值
    p_hash = ''
    for i in range(8):
        for j in range(8):
            if dct[i, j] > avg:
                p_hash += '1'
            else:
                p_hash += '0'
    return p_hash

def compare_average_hash(hash1, hash2):
    # 计算汉明距离
    distance = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            distance += 1
    return distance
if __name__ == '__main__':
    img1 = cv2.imread('lenna.png')
    img2 = cv2.imread('lenna_gaussian_blur.png')
    a_hash1 = cal_average_hash(img1)
    a_hash2 = cal_average_hash(img2)
    print(a_hash1)
    print(a_hash2)
    distance = compare_average_hash(a_hash1, a_hash2)
    print("均值哈希差异为：",distance)
    d_hash1 = cal_diff_hash(img1)
    d_hash2 = cal_diff_hash(img2)
    print(d_hash1)
    print(d_hash2)
    distance = compare_average_hash(d_hash1, d_hash2)
    print("差值哈希差异为：",distance)
    p_hash1 = cal_p_hash(img1)
    p_hash2 = cal_p_hash(img2)
    print(p_hash1)
    print(p_hash2)
    distance = compare_average_hash(p_hash1, p_hash2)
    print("感知哈希差异为：",distance)
