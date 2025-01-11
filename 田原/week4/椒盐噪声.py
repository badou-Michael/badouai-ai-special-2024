import numpy as np
import cv2
import random

def fun1(src, percentage):

    NoiseImg = src.copy()  # 使用副本以避免修改原始图像
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])

    # 随机选择噪声点的位置
    rows, cols = src.shape
    indices = np.random.randint(0, rows * cols, size=NoiseNum)
    randX = indices // cols
    randY = indices % cols

    # 随机分配椒盐噪声
    for i in range(NoiseNum):
        if random.random() <= 0.5:  # 椒噪声（黑点）
            NoiseImg[randX[i], randY[i]] = 0
        else:  # 盐噪声（白点）
            NoiseImg[randX[i], randY[i]] = 255

    return NoiseImg


# 读取灰度图像
img = cv2.imread('lenna.png', 0)
img_noisy = fun1(img, 0.8)


# 显示原始和加噪后的图像
cv2.imshow('Original Image', img)
cv2.imshow('PepperandSalt Image', img_noisy)
cv2.waitKey(0)
cv2.destroyAllWindows()
