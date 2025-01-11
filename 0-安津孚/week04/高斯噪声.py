import numpy as np
import cv2
from numpy import shape
import random


def add_gaussian_noise(src, mean, sigma, percentage):
    noise_img = src.copy()
    num_pixels = int(percentage * src.size)
    for _ in range(num_pixels):
        x = random.randint(0, src.shape[1] - 1)
        y = random.randint(0, src.shape[0] - 1)
        noise_value = random.gauss(mean, sigma)
        # 确保像素值在0到255之间
        noise_img[y, x] = np.clip(noise_img[y, x] + noise_value, 0, 255)
    return noise_img


img = cv2.imread('../week02/lenna.png', 0)
img1 = add_gaussian_noise(img, 3, 8, 0.6)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source', img2)
cv2.imshow('lenna_GaussianNoise', img1)
cv2.waitKey(0)
