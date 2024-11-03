# 1 高斯噪声，椒盐噪声手写实现
import numpy as np
import cv2
import random


def GaussianNoise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):

        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


def pepperNoise(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):

        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg

img = cv2.imread('lenna.png', 0)
img1 = GaussianNoise(img, 2, 4, 0.8)
img2 = pepperNoise(img, 0.8)
