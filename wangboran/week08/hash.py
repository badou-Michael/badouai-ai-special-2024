#-*- coding:utf-8 -*-
# author: 王博然
import cv2
import numpy as np

# Average Hash Algorithm
def aHash(img):
    row = 8
    col = 8
    total = 0
    avg = 0
    hash_str = ''
    # 缩放
    img = cv2.resize(img, (col, row), interpolation=cv2.INTER_CUBIC)
    # 像素和
    for i in range(row):
        for j in range(col):
            total += img[i,j]
    avg = total/(row * col)
    for i in range(row):
        for j in range(col):
            if img[i,j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

# Difference Hash Algorithm
def dHash(img):
    row = 8
    col = 9
    hash_str = ''
    # 缩放
    img = cv2.resize(img, (col, row), interpolation=cv2.INTER_CUBIC)
    # 每行前一个像素大于后一个像素为1，相反为0
    for i in range(row):
        for j in range(col - 1):
            if img[i,j] > img[i,j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

# Hamming Distance
def hmDistance(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n

if __name__ == '__main__':
    # 直接读入灰度图片
    img_src = cv2.imread('../lenna.png', cv2.IMREAD_GRAYSCALE)
    img_noise = cv2.GaussianBlur(img_src, (5, 5), 0)

    # cv2.imshow('src', img_src)
    # cv2.imshow('noise', img_noise)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 均值哈希
    hash1 = aHash(img_src)
    hash2 = aHash(img_noise)
    n = hmDistance(hash1, hash2)
    print('均值哈希算法相似度: ',n)

    # 差值哈希
    hash1 = dHash(img_src)
    hash2 = dHash(img_noise)
    n = hmDistance(hash1, hash2)
    print('差值哈希算法相似度: ',n)