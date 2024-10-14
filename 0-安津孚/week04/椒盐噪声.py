import numpy as np
import cv2
from numpy import shape
import random


def salted_pepper_noise(src, percentage):
    noise_img = src.copy()
    num_pixels = int(percentage * src.size)
    for _ in range(num_pixels):
        x = random.randint(0, src.shape[1] - 1)
        y = random.randint(0, src.shape[0] - 1)
        # 确保像素值在0到255之间
        if random.random() <= 0.5:
            noise_img[x, y] = 0
        else:
            noise_img[x, y] = 255
    return noise_img


def salted_pepper_noise2(src, percentage):
    noise_img = src.copy()
    print(src.size)
    num_pixels = int(percentage * src.size)
    num_rows, num_cols = src.shape[:2]  # 获取图像的行和列数

    # 生成随机的坐标点
    coords = np.random.choice(num_rows * num_cols, num_pixels, replace=False)
    rows = coords // num_cols
    cols = coords % num_cols

    # 生成随机的噪声值
    noise_values = np.random.choice([0, 255], size=num_pixels, p=[0.5, 0.5])

    # 应用噪声
    noise_img[rows, cols] = noise_values

    # 确保像素值在0到255之间
    noise_img = np.clip(noise_img, 0, 255)
    return noise_img


img = cv2.imread('../week02/lenna.png', 0)
img1 = salted_pepper_noise(img, 0.6)
# 在文件夹中写入命名为lenna_PepperandSalt.png的加噪后的图片
# cv2.imwrite('lenna_PepperandSalt.png',img1)

img = cv2.imread('../week02/lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', img2)
cv2.imshow('lenna_PepperandSalt', img1)
cv2.waitKey(0)
