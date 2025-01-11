import random

import numpy as np
import cv2


# 方法1
def gauss_noise_1(img, sigma, mean, ratio):
    gauss_img = img.copy()
    # 原图片大小
    total_pixelNum = img.shape[0] * img.shape[1]
    # 计算高斯噪声的范围
    gaussNum = int(ratio * total_pixelNum)
    # 一次性生成高斯噪声
    noise = np.random.normal(mean, sigma, gaussNum)
    for i in range(gaussNum):
        # 随机选取坐标
        x = np.random.randint(0, img.shape[0])
        y = np.random.randint(0, img.shape[1])
        # 加噪声
        gauss_img[x][y] += noise[i]
        if gauss_img[x][y] > 255:
            gauss_img[x][y] = 255
        elif gauss_img[x][y] < 0:
            gauss_img[x][y] = 0
    return gauss_img


# 方法2
def gauss_noise_2(img, sigma, mean, ratio):
    gauss_img = img.copy()
    # 原图片大小
    total_pixelNum = img.shape[0] * img.shape[1]
    # 计算高斯噪声的范围
    gaussNum = int(ratio * total_pixelNum)
    # 一次性生成高斯噪声
    noise = np.random.normal(mean, sigma, gaussNum)
    # 获取图片所有坐标
    all_coords = [(x, y) for x in range(img.shape[0]) for y in range(img.shape[1])]
    print('所有坐标的总长度位:', len(all_coords))
    # 获取要添加噪声的坐标     参数1:总长度  参数2:选取个数  参数3:是否可以重复  返回值:索引数组
    index_array = np.random.choice(len(all_coords), size=gaussNum, replace=False)
    # 遍历索引数组，将噪声添加到对应坐标
    for i in range(len(index_array)):
        x, y = all_coords[i]
        # 加噪声
        gauss_img[x][y] += noise[i]
        # 防止溢出
        if gauss_img[x][y] > 255:
            # 超过255，取255
            gauss_img[x][y] = 255
        elif gauss_img[x][y] < 0:
            # 小于0，取0
            gauss_img[x][y] = 0
    return gauss_img

#方法3
def gauss_noise_3(img, sigma, mean, ratio):
    gauss_img = img.copy()
    # 原图片大小
    total_pixelNum = img.shape[0] * img.shape[1]
    # 计算高斯噪声的范围
    gaussNum = int(ratio * total_pixelNum)
    # 一次性生成高斯噪声
    noise = np.random.normal(mean, sigma, gaussNum)
    for i in range(gaussNum):
        # 随机选取坐标
        x = np.random.randint(0, img.shape[0])
        y = np.random.randint(0, img.shape[1])
        # 加噪声
        gauss_img[x][y] += random.gauss(mean, sigma)
        if gauss_img[x][y] > 255:
            gauss_img[x][y] = 255
        elif gauss_img[x][y] < 0:
            gauss_img[x][y] = 0
    return gauss_img


img = cv2.imread('lenna.png', 0)
gauss_noise_img = gauss_noise_1(img, 10, 0, 0.8)
cv2.imshow('gauss_noise', gauss_noise_img)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('original', img2)
cv2.waitKey(0)
