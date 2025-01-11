import numpy as np
import cv2
from numpy import shape
import random


def noise(src, means, sigma, percent):
    img = src

    rangeNum = int(src.shape[0] * src.shape[1] * percent)
    for i in range(rangeNum):
        randomX = random.randint(0, src.shape[0] - 1)
        randomY = random.randint(0, src.shape[1] - 1)
        img[randomX, randomY] = img[randomX, randomY] + random.gauss(means, sigma)
        print(img[randomX, randomY])
        if img[randomX, randomY] > 255:
            img[randomX, randomY] = 255
        elif img[randomX, randomY] < 0:
            img[randomX, randomY] = 0

    return img


orgin = cv2.imread("lenna.png", 0)
imgGauss = noise(orgin, 2, 4, 0.8)
cv2.imshow("gauss", imgGauss)
cv2.waitKey(0)
