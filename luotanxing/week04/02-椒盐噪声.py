import numpy as np
import cv2
from numpy import shape
import random


def noise(src,  percent):
    w, h = src.shape
    points_number = int(percent * w * h)
    for i in range(points_number):
        rand_x = random.randint(0, w - 1)
        rand_y = random.randint(0, h - 1)
        if random.random() < 0.5:
            src[rand_x, rand_y] = 0
        else:
            src[rand_x, rand_y] = 255
    return src


if __name__ == '__main__':
    img = cv2.imread('../week02/lenna.png')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = noise(img2,  0.5)
    cv2.imshow('noise', img_gauss)
    cv2.waitKey(0)
