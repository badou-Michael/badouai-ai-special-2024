# -*- coding: utf-8 -*-
# time: 2024/10/17 17:04
# file: PeolperSalt.py
# author: flame
import numpy as np
import cv2
from numpy import shape
import random

# 定义一个函数，用于向图像中添加椒盐噪声
def fun1(src, percentage):
    """
    向图像中添加椒盐噪声。

    参数:
        src: 输入的原始图像。
        percentage: 添加噪声的像素比例。

    返回:
        添加噪声后的图像。
    """
    ## 函数逻辑梳理
    # 1. 将原始图像赋值给NoiseImg，准备添加噪声。
    # 2. 根据图像尺寸和指定的噪声比例计算需要添加的噪声数量。
    # 3. 遍历每个需要添加噪声的像素，随机选择一个像素位置。
    # 4. 随机决定此像素是添加椒噪声（0）还是盐噪声（255）。
    # 5. 返回添加噪声后的图像。

    # 将原始图像赋值给NoiseImg，准备添加噪声
    NoiseImg = src

    # 计算需要添加的噪声数量，根据图像尺寸和指定的噪声比例
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])

    # 遍历每个需要添加噪声的像素
    for i in range(NoiseNum):
        # 随机选择一个像素的x坐标
        randx = random.randint(0, src.shape[0] - 1)

        # 随机选择一个像素的y坐标
        randy = random.randint(0, src.shape[1] - 1)

        # 随机决定此像素是添加椒噪声（0）还是盐噪声（255）
        if random.random() <= 0.5:
            # 如果随机数小于等于0.5，将该像素设置为0（椒噪声）
            NoiseImg[randx, randy] = 0
        else:
            # 否则，将该像素设置为255（盐噪声）
            NoiseImg[randx, randy] = 255

    # 返回添加噪声后的图像
    return NoiseImg

# 读取图像，转换为灰度图
img = cv2.imread("lenna.png", 0)

# 添加椒盐噪声，噪声像素比例为0.8
img1 = fun1(img, 0.8)

# 读取图像，保持原始颜色
img = cv2.imread("lenna.png")

# 将图像转换为灰度图
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示原始灰度图和添加噪声后的图像
cv2.imshow("1_2", np.hstack([img2, img1]))

# 等待用户按键，显示图像
cv2.waitKey()
