import numpy as np
import cv2
from numpy import shape
import random

#高斯噪声的图片加噪

def gaussNoise(src, mu, sigma, percentage):
    imgGauss = src
    gaussNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(gaussNum):
        randomX = random.randint(0, src.shape[0] - 1)
        randomY = random.randint(0, src.shape[1] - 1)

        # print(random.gauss(mu,sigma),",",imgGauss[randomX][randomY])
        imgGauss[randomX][randomY] = imgGauss[randomX][randomY] + random.gauss(mu, sigma)

        if imgGauss[randomX][randomY] < 0:
            imgGauss[randomX][randomY] = 0
        elif imgGauss[randomX][randomY] > 255:
            imgGauss[randomX][randomY] = 255
    return imgGauss


img0 = cv2.imread("lenna.png", 0)

img = cv2.imread("lenna.png", 0)
imgGauss = gaussNoise(img, 2, 15, 1.0)

cv2.imshow("img", img0)
cv2.imshow("imgGauss", imgGauss)
# cv2.imwrite("imgGauss.png", imgGauss)
cv2.waitKey(0)
