import cv2
import random


def noise(src, percent):
    rangeNum = int(percent * src.shape[0] * src.shape[1])
    for i in range(rangeNum):
        randomX = random.randint(0, src.shape[0] - 1)
        randomY = random.randint(0, src.shape[1] - 1)
        if src[randomX, randomY] < 128:
            src[randomX, randomY] = 0
        else:
            src[randomX, randomY] = 255

    return src


orgin = cv2.imread("lenna.png", 0)
img1 = noise(orgin, 0.8)
cv2.imshow("椒盐", img1)
cv2.waitKey(0)
