import numpy as np
import cv2
import random

def salt_and_pepper(img, percentage):
    Noise_img = img.copy()
    Noise_num = int(img.shape[0] * img.shape[1] * percentage)
    for i in range(Noise_num):
        x = random.randint(0, img.shape[0]-1)
        y = random.randint(0, img.shape[1]-1)
        
        if random.randint(0, 1) > 0.5:
            Noise_img[x][y] = 255
        else:
            Noise_img[x][y] = 0
    return Noise_img
        
        
img = cv2.imread('lenna.png', 0)
img_noise = salt_and_pepper(img, 0.8)
img2 = cv2.imread('lenna.png')
imggray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img_noise)
cv2.imshow('imggray', imggray)
cv2.waitKey(0)

