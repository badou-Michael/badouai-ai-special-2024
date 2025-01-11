import cv2
import numpy as np


# 均值哈希算法
def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 求像素均值
    avg = np.mean(gray)
    # hash值拼接
    hash = ""
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash += "1"
            else:
                hash += "0"
    return hash


# 差值哈希算法
def dHash(img):
    # 缩放成8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 不需要计算均值，直接比较每一行前一个和后一个的大小
    hash = ""
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash += "1"
            else:
                hash += "0"
    return hash


# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


img1 = cv2.imread('../week02/lenna.png')
img2 = cv2.imread('../week04/lenna_noise.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', n)
