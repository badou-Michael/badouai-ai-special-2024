import numpy as np
import cv2
import random

# src: 原图, mu、sigma高斯分布参数, percentage: 加躁比例
# return: 加躁图
def gaussianNoise(src, mu, sigma, percentage):
    noise_img = src.copy()  # 深拷贝, 避免修改src的值
    noise_num = int(percentage * src.shape[0] * src.shape[1])
    for i in range(noise_num):
        rand_x = random.randint(0, src.shape[0] - 1)
        rand_y = random.randint(0, src.shape[1] - 1)
        noise_img[rand_x][rand_y] = noise_img[rand_x][rand_y] + random.gauss(mu, sigma)
        if noise_img[rand_x][rand_y] < 0:
            noise_img[rand_x][rand_y] = 0
        elif noise_img[rand_x][rand_y] > 255:
            noise_img[rand_x][rand_y] = 255
    return noise_img

src_gray = cv2.imread('../lenna.png', 0)
out_gray = gaussianNoise(src_gray, 2, 10, 0.8)
cv2.imshow('source', src_gray)
cv2.imshow('gaussian_lenna', out_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()