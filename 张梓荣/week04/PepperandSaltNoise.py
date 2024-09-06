import random

import cv2
import numpy as np


def process(img, persent):
    w, h, channels = img.shape[:3]
    cycle = int(persent * h * w)
    print(cycle, channels)
    distImg = img.copy()
    for channel in range(channels):
        for c in range(cycle):
            randomX = random.randint(0, w - 1)
            randomY = random.randint(0, h - 1)
            if random.random() < 0.5:
                distImg[randomX, randomY, channel] = 0
            else:
                distImg[randomX, randomY, channel] = 255
    return distImg


img = cv2.imread('lenna.png')
distImg = process(img, 0.6)
cv2.imshow("souce", img)
cv2.imshow("distImg", distImg)
cv2.waitKey(0)
