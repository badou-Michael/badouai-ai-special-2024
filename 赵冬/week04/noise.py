import random

import cv2
import numpy as np
from skimage import util

# 高斯噪声
def gaussian_noise(src, means, sigma, percentage):
    dst_img = src
    src_w, src_h = src.shape
    num = int(percentage * src_w * src_h)
    for i in range(num):
        random_x = random.randint(0, src_w - 1)
        random_y = random.randint(0, src_h - 1)
        dst_img[random_x, random_y] = dst_img[random_x, random_y] + random.gauss(means, sigma)
        if dst_img[random_x, random_y] < 0:
            dst_img[random_x, random_y] = 0
        elif dst_img[random_x, random_y] > 255:
            dst_img[random_x, random_y] = 255
    return dst_img


# 椒盐噪声
def peper_salt_noise(src, percentage):
    dst_img = src
    src_w, src_h = src.shape
    num = int(percentage * src_w * src_h)
    for i in range(num):
        random_x = random.randint(0, src_w - 1)
        random_y = random.randint(0, src_h - 1)
        if random.random()<=0.5:
            dst_img[random_x, random_y] = 0
        else:
            dst_img[random_x, random_y] = 255
    return dst_img


if __name__ == '__main__':
    # 实现高斯噪声、椒盐噪声手写实现
    img1 = cv2.imread('lenna.png', 0)
    gau1 = gaussian_noise(img1, 2, 10, 0.8)
    img2 = cv2.imread('lenna.png', 0)
    peper2 = peper_salt_noise(img2, 0.2)
    source = cv2.imread('lenna.png', 0)
    cv2.imshow('source', np.hstack([source, gau1, peper2]))
    cv2.waitKey(0)

    #噪声接口调用
    gau3 = util.random_noise(cv2.imread('lenna.png'), mode="gaussian", mean=0, var = 0.01)
    sp4 = util.random_noise(cv2.imread('lenna.png'), mode="s&p", amount=0.5)
    source = cv2.imread('lenna.png')
    cv2.imshow('source', gau3)
    cv2.waitKey(0)
    cv2.imshow('source', sp4)
    cv2.waitKey(0)
