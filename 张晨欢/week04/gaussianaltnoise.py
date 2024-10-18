import cv2
import numpy as np
from numpy import random


#高斯噪声

def GaussianNoise(image1, means, sigma, percentage):
    # 设置需要处理的像素个数
    num_pixels_to_process = int(percentage * image1.shape[0] * image1.shape[1])
    # 获取图像的高度和宽度
    height, width = image1.shape
    # 遍历需要处理的像素个数
    for _ in range(num_pixels_to_process):
        img = image1
        # 随机选择一个像素位置
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)

        # 生成一个高斯随机数
        gaussian_noise = random.normal(means, sigma)
        # 在选定的像素位置上加上高斯随机数
        img[y, x] = np.clip(image[y, x] + gaussian_noise, 0, 255)
        if img[y, x] < 0:
            img[y, x] = 0
        elif img[y, x] > 255:
            img[y, x] = 255

    return img



#椒盐噪声

def add_salt_pepper_noise(image2, percentage):
    # 设置需要处理的像素个数
    num_pixels_to_process = int(percentage * image2.shape[0] * image2.shape[1])
    height, width = image2.shape
    for i in range(num_pixels_to_process):
        img = image2
        # 随机选择一个像素位置
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        if random.random() <= 0.5:
            img[x, y] = 0
        else:
            img[x, y] = 255
    return img


if __name__ == "__main__":
    # 读取图像
    image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('source', image)
    img1 = GaussianNoise(image, 2, 50, 0.9)
    img1=add_salt_pepper_noise(image, 0.1)
    cv2.imshow('lenna_GaussianNoise', img1)
    cv2.waitKey(0)
