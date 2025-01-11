import numpy as np
import cv2
import random


# 定义椒盐噪声函数  方法1
def impulse_noise(img, ratio):
    impulse_img = img.copy()
    # 获取所有坐标
    total_pixel = img.shape[0] * img.shape[1]
    # 获取椒盐噪声的数量
    salt_num = int(total_pixel * ratio)
    #循环遍历需要添加椒盐噪声的坐标
    for i in range(salt_num):
        x = np.random.randint(0, img.shape[0])
        y = np.random.randint(0, img.shape[1])
        if random.random() >= 0.5:
            impulse_img[x][y] = 255
        else:
            impulse_img[x][y] = 0
    return impulse_img


# 方法2
def impulse_noise_2(img, ratio):
    impulse_img = img.copy()
    #获取所有坐标
    total_pixel = img.shape[0] * img.shape[1]
    #获取椒盐噪声的数量
    salt_num = int(total_pixel * ratio)
    #获取所有坐标
    all_coords = [(x, y) for x in range(img.shape[0]) for y in range(img.shape[1])]
    #随机选取salt_num个坐标  保证坐标不重复
    index_array = np.random.choice(len(all_coords), salt_num, False)
    for i in index_array:
        x, y = all_coords[i]
        if random.random() >= 0.5:
            impulse_img[x][y] = 255
        else:
            impulse_img[x][y] = 0
    return impulse_img


img = cv2.imread('lenna.png', 0)
impulse_img = impulse_noise(img, 0.8)
impulse_img2 = impulse_noise_2(img, 0.8)
cv2.imshow('impulse_img', np.hstack([impulse_img,impulse_img2]))
cv2.imshow('img', img)
cv2.waitKey(0)
