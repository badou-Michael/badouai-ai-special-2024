import cv2
import numpy as np
import random

# 均值哈希算法
def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s += gray[i, j]
    avg = s / 64
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

# 差值哈希算法
def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

# 哈希值对比
def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n

# 向图像添加高斯噪声
def add_noise(img, mean=0, sigma=25):
    """
    向图像中添加高斯噪声
    :param img: 输入的图像
    :param mean: 噪声的均值，通常为0
    :param sigma: 噪声的标准差，控制噪声强度
    :return: 添加噪声后的图像
    """
    row, col, ch = img.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.uint8(np.clip(img + gauss, 0, 255))
    return noisy

# 读取原图
img1 = cv2.imread('lenna.png')

# 添加噪声生成新图像
img2 = add_noise(img1)

# 计算均值哈希值
hash1 = aHash(img1)
hash2 = aHash(img2)
print("均值哈希值（原图）：", hash1)
print("均值哈希值（噪声图）：", hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)
# 越大越不相似

# 计算差值哈希值
hash1 = dHash(img1)
hash2 = dHash(img2)
print("差值哈希值（原图）：", hash1)
print("差值哈希值（噪声图）：", hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', n)

