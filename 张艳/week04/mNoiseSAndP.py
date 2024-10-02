import numpy as np
import cv2
from numpy import shape
import random

#椒盐噪声的图片加噪

def saltpeppernoise(img, percentage):
    imgSP = img.copy()
    # 区分：
    # imgSP = img.copy() # 创建副本，不改变img
    # imgSP2 = img # 不创建副本，改变img
    spNum = int(percentage * img.shape[0] * img.shape[1])
    for i in range(spNum):
        randomX = random.randint(0, img.shape[0] - 1)
        randomY = random.randint(0, img.shape[1] - 1)
        if random.random() < 0.5:
            imgSP[randomX, randomY] = 0
        else:
            imgSP[randomX, randomY] = 255
    return imgSP


img = cv2.imread("lenna.png", 0)
imgSP2 = saltpeppernoise(img, 0.01)

cv2.imshow("img", img)
cv2.imshow("imgSP", imgSP2)
cv2.imwrite("imgSP.png", imgSP2)
cv2.waitKey(0)
