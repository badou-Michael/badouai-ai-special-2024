import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

# 灰度化
img = cv2.imread("lenna.png")
height, width = img.shape[:2]

img_gray = np.zeros([height, width], img.dtype)
for i in range(height):
    for j in range(width):
        rgb = img[i, j]
        img_gray[i, j] = int(rgb[0] * 0.11 + rgb[1] * 0.59 + rgb[2] * 0.3)

img_gray = rgb2gray(img)

hang, shu = img_gray.shape
for i in range(hang):
    for j in range(shu):
        if (img_gray[i, j] < 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1

img_binary = np.where(img_gray < 0.5, 0, 1)
