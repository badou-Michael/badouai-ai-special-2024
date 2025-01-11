import numpy as np
import cv2
from matplotlib import pyplot as plt


def function(image):
    height, width, channels = image.shape
    void_image = np.zeros((800, 800, channels), dtype=np.uint8)
    print(void_image)
    hh = 800 / height
    wh = 800 / width
    for i in range(800):
        for j in range(800):
            x = int(i / hh + 0.5)
            y = int(j / wh + 0.5)
            void_image[i ,j] = image[x, y]
    return void_image

image = cv2.imread('moon.JPG')
print(image.shape)
void_image = function(image)

# void_image1 = cv2.resize(image,(800, 800), interpolation=cv2.INTER_NEAREST)
# void_image = cv2.resize(image,(800, 800), interpolation=cv2.INTER_LINEAR)
cv2.imshow("nearest interpolation", void_image)
# cv2.imshow('original image', void_image)
cv2.waitKey(0)
