import random

import cv2
import numpy as np


def process(img, mean, sigma, persent):
    w, h, channels = img.shape[:3]
    cycle = int(persent * h * w)
    distImg = img.copy()
    for channel in range(channels):
        for c in range(cycle):
            randomX = random.randint(0, w - 1)
            randomY = random.randint(0, h - 1)
            distImg[randomX, randomY, channel] = int(img[randomX, randomY, channel] + random.gauss(mean, sigma))
            if distImg[randomX, randomY, channel] > 255:
                distImg[randomX, randomY, channel] = 255
            elif distImg[randomX, randomY, channel] < 0:
                distImg[randomX, randomY, channel] = 0
    return distImg


img = cv2.imread('lenna.png')
distImg = process(img, 2, 4, 1)
cv2.imshow("souce", img)
cv2.imshow("distImg", distImg)
cv2.waitKey(0)
