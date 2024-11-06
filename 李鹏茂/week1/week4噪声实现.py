import random
from skimage import util
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaosi(img,mean, sigma, perser):
    height, width, channels = img.shape
    print(height, width, channels)
    haxi_map = {}
    num = width * height * perser
    result_img = img.copy()  # 创建结果图像

    i = 0
    while i < num:
        w = np.random.randint(0, width)
        h = np.random.randint(0, height)
        key = (h, w)

        if key not in haxi_map:
            haxi_map[key] = True  # 存储唯一的坐标
            gamma = random.gauss(mean, sigma)
            for c in range(channels):
                # 更新像素值，确保范围在[0, 255]
                result_img[h, w, c] = np.clip(result_img[h, w, c] + gamma, 0, 255)
            print(key)
            i += 1  # 仅在找到新坐标时增加计数

    return result_img  # 返回结果图像

def jiaoyan(img,perser):
    height, width, channels = img.shape
    print(height, width, channels)
    haxi_map = {}
    num = width * height * perser
    result_img = img.copy()  # 创建结果图像

    i = 0
    while i < num:
        w = np.random.randint(0, width-1)
        h = np.random.randint(0, height-1)
        key = (h, w)

        if key not in haxi_map:
            haxi_map[key] = True  # 存储唯一的坐标
            jiaoyan = random.random()
            if jiaoyan < 0.5:
                jiaoyan = 0
            else:
                jiaoyan = 255
            for c in range(channels):
                # 更新像素值，确保范围在[0, 255]
                result_img[h, w, c] =  jiaoyan
            print(key)
            i += 1  # 仅在找到新坐标时增加计数

    return result_img  # 返回结果图像



if __name__ == '__main__':
    image = cv2.imread(r'C:\Users\Lenovo\Desktop\meinv.png')


    imggaosi=gaosi(image,2,4,0.5)
    #高斯噪声 参数0.8 会很慢怎么改进？
    imgjiaoyan=jiaoyan(image,0.2)

    # 添加高斯噪声
    gaussian_noisy_image = util.random_noise(image, mode='gaussian', var=0.01)
    # 使用均值滤波去噪
    # mean_denoised_image = cv2.blur(gaussian_noisy_image, (3, 3))
    mean_denoised_image = cv2.GaussianBlur(gaussian_noisy_image, (5, 5),1)
    # 用这个效果更好因为高斯滤波越近权重越大
    # 添加椒盐噪声
    salt_and_pepper_noisy_image = util.random_noise(image, mode='s&p', amount=0.2)
    # #中值去椒盐
    # 转换为uint8类型
    salt_and_pepper_noisy_image = (salt_and_pepper_noisy_image * 255).astype(np.uint8)
    salt_denoised_image = cv2.medianBlur(salt_and_pepper_noisy_image, ksize=3)

    cv2.imshow('gaosi', imggaosi)
    cv2.imshow('jiaoyan', imgjiaoyan)
    cv2.imshow('jiekogaosi', gaussian_noisy_image)
    cv2.imshow('jiekojiaoyan', salt_and_pepper_noisy_image)
    cv2.imshow('den_jiekogaosi', mean_denoised_image)
    cv2.imshow('den_jiekojiaoyan', salt_denoised_image)
    cv2.imshow('yuantu',image)
    cv2.waitKey()