

import cv2
import numpy as np


# 均值哈希
def mean_hash(img, n):
    # 1.缩放为 n*n
    img = cv2.resize(img, (n, n), interpolation=cv2.INTER_CUBIC)
    # 2.转为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3.计算：像素累加值
    s = sum(img_gray[i, j] for i in range(n) for j in range(n))
    # 4.求平均灰度
    avg = s / (n*n)
    # 5.计算：哈希值字符串
    hash_str = ''.join('1' if img_gray[i, j] > avg else '0' for i in range(n) for j in range(n))
    return hash_str


# 差值哈希
def diff_hash(img, n):
    # 1.缩放 n*(n+1)
    img = cv2.resize(img, (n+1, n), interpolation=cv2.INTER_CUBIC)
    # 2.转换灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3.计算：哈希值字符串
    hash_str = ''.join('1' if img_gray[i, j] > img_gray[i, j+1] else '0' for i in range(n) for j in range(n))
    return hash_str


# 哈希值对比
def cmp_hash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1

    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n


img_1 = cv2.imread('lenna.png')
img_2 = cv2.imread('lenna_PepperandSalt.png')
hash_1 = mean_hash(img_1, 8)
hash_2 = mean_hash(img_2, 8)
print(hash_1)
print(hash_2)
n = cmp_hash(hash_1, hash_2)
print('均值哈希算法相似度：', n)


hash_1 = diff_hash(img_1, 8)
hash_2 = diff_hash(img_2, 8)
print(hash_1)
print(hash_2)
n = cmp_hash(hash_1, hash_2)
print('差值哈希算法相似度：', n)

