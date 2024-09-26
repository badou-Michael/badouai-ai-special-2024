#[模块2:椒盐噪声及泊松噪声] 一: 根据原理自定义函数实现
import numpy as np
import cv2 as cv
import random
def sp_noise(src, snr):     # 传入图像及噪声比例
    h = src.shape[0]
    w = src.shape[1]
    sp = h * w              # 计算输入图像像素总个数
    np = int(sp * snr)      # 计算要加噪的像素个数
    img_noise = src
    for num in range(np):
        X = random.randint(0, h - 1)
        Y = random.randint(0, w - 1)
        if random.random() >= 0.5:
            img_noise[X, Y] = 255
        else:
            img_noise[X, Y] = 0
    return img_noise

if __name__ == "__main__":
    img =cv.imread("E:/GUO_APP/GUO_AI/picture/lenna.png")
    img1 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # img1 = cv.imread("E:/GUO_APP/GUO_AI/picture/lenna.png", 0)  # 读取lenna的灰度图
    cv.imshow('img', img)
    cv.imshow('img1', img1)
    SpNoise = sp_noise(img1, 0.2)
    cv.imshow('sp_noise', SpNoise)
    cv.waitKey(0)

# [模块2:椒盐噪声及泊松噪声] 二: 通过调用skimage.util.noise()函数

import skimage.util as ut
from matplotlib import pyplot as plt

img = plt.imread("E:/GUO_APP/GUO_AI/picture/lenna.png")
img_s_noise = ut.random_noise(img,mode='salt',amount=0.3)    # 添加盐噪声
img_p_noise = ut.random_noise(img,mode='pepper',amount=0.3)  # 添加椒噪声
img_sp_noise = ut.random_noise(img,mode='s&p',amount=0.3,salt_vs_pepper=0.7)   # 添加椒盐噪声

img_s_noise = ut.random_noise(img,mode='poisson')    # 添加泊松噪声
plt.rcParams['font.sans-serif'] = ['SimHei']    #将plt.title内的中文字体设置为黑体
plt.subplot(331)
plt.title('输入图像img')
plt.imshow(img)
plt.subplot(332)
plt.title('盐噪声图:噪声比0.3')
plt.imshow(img_s_noise)
plt.subplot(333)
plt.title('椒噪声图:噪声比0.3')
plt.imshow(img_p_noise)
plt.subplot(334)
plt.title('椒盐噪声图:噪声比0.3,盐椒噪声比0.7')
plt.imshow(img_sp_noise)
plt.subplot(335)
plt.title('泊松噪声图')
plt.imshow(img)
plt.show()
