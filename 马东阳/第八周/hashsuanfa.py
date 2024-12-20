# 哈希算法

import cv2
import numpy as np

# 均值哈希
def junzhiHash(img):
    # 缩放8*8
    img = cv2.resize(img, (8,8),interpolation=cv2.INTER_CUBIC)
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 赋初值
    s = 0
    hash_s = ''
    # 遍历累加求像素之和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 平均灰度
    avg = s/64
    # 灰度大于平均值为1，否则为0，生成图片的哈希值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_s = hash_s + '1'
            else:
                hash_s = hash_s +'0'
    return hash_s

# 差值算法
def chazhiHash(img):
    # 缩放8*9
    img = cv2.resize(img, (9,8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_s = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_s = hash_s + '1'
            else:
                hash_s = hash_s +'0'
    return hash_s

# 哈希值对比
def bijiao(hash1, hash2):
    n = 0
    # hash长度不同则返回-1, 传参出错
    if len(hash1) != len(hash2):
        return -1
    # 如果不相等，则n计数+1，n最终为相似度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('noise_gs_img.jpg')  # 高斯噪声lenna图片
hash1 = junzhiHash(img1)
hash2 = junzhiHash(img2)
print(hash1)
print(hash2)
n=bijiao(hash1,hash2)
print('均值哈希算法相似度：',n)

hash3= chazhiHash(img1)
hash4= chazhiHash(img2)
print(hash3)
print(hash4)
n=bijiao(hash3,hash4)
print('差值哈希算法相似度：',n)
