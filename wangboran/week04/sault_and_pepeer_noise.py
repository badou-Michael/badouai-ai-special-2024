import numpy as np
import cv2
import random

# src: 原图, percentage: 加躁比例
# return: 加躁图
def saultPepeerNoise(src, percentage):
    noise_img = src.copy()  # 深拷贝, 避免修改src的值
    noise_num = int(percentage * src.shape[0] * src.shape[1])
    for i in range(noise_num):
        rand_x = random.randint(0, src.shape[0] - 1)
        rand_y = random.randint(0, src.shape[1] - 1)
        # random.random()‌ 函数用于生成一个[0.0, 1.0)之间的随机浮点数
        if random.random() <= 0.5:
            noise_img[rand_x][rand_y] = 0
        else:
            noise_img[rand_x][rand_y] = 255
    return noise_img

src_gray = cv2.imread('../lenna.png', 0)
out_gray = saultPepeerNoise(src_gray, 0.2)
cv2.imshow('source', src_gray)
cv2.imshow('pepperAndSalt_lenna', out_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()