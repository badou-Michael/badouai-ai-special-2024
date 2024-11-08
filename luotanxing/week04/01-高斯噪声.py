import numpy as np
import cv2
from numpy import shape
import random


def gauss_noise(src, means, sigma, percent):
    w, h = src.shape
    IMAGE_SOURCE = src
    points_number = int(percent * w * h)
    for i in range(points_number):
        rand_x = random.randint(0, w - 1)
        rand_y = random.randint(0, h - 1)
        IMAGE_SOURCE[rand_x, rand_y] = IMAGE_SOURCE[rand_x, rand_y] + random.gauss(means, sigma)
        if IMAGE_SOURCE[rand_x, rand_y] < 0:
            IMAGE_SOURCE[rand_x, rand_y] = 0
        elif IMAGE_SOURCE[rand_x, rand_y] > 255:
            IMAGE_SOURCE[rand_x, rand_y] = 255
    return IMAGE_SOURCE

if __name__ == '__main__':
    img = cv2.imread('../week02/lenna.png')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = gauss_noise(img2,2,4,0.8)
    cv2.imshow('lenna_GaussianNoise', img_gauss)
    cv2.waitKey(0)