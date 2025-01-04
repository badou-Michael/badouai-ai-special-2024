#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：hashtest.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/11/14 12:11
'''
import cv2

# 均值哈希算法：缩放 灰度图 求均值 生成哈希值
def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 初始化像素和&哈希值
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    average = s / 64
    for i in range(8):
        for j in range(8):
            if gray[i, j] > average:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 差值哈希算法
def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)   # cv2.resize由于调整图像大小时格式默认设置为(width,height)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 初始化哈希值
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 用汉明距离度量图像之间的相似度
def cmpHash(hash1, hash2):
    n = 0   # 初始化哈希值不同的点的个数
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    p = n / 64
    return p


img1 = cv2.imread("lenna.png")
img2 = cv2.imread("lenna_noise.png")
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
p = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', p)

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
p = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', p)