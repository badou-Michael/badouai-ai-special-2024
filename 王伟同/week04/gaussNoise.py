import random

import cv2
import numpy as np


def gaussNoise(src, mean, variance, percentage):
    gauss_picture = np.copy(src).astype(np.float64)
    size = int(percentage * len(gauss_picture[0]) * len(gauss_picture[1]))
    for i in range(size):
        random_x = random.randint(0, len(gauss_picture[0]) - 1)
        random_y = random.randint(0, len(gauss_picture[1]) - 1)
        gauss_picture[random_x, random_y] += random.gauss(mean, variance)
    gauss_picture = np.clip(gauss_picture, 0, 255).astype(np.uint8)
    return gauss_picture

src = cv2.imread("lenna.png", 0)
gauss_picture = gaussNoise(src, 25, 4, 1)
src1 = cv2.imread("lenna.png")
src1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
cv2.imshow("original", src1)
cv2.imshow("gauss", gauss_picture)
cv2.waitKey()
