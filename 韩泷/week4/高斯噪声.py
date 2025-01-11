import numpy as np
import cv2
import random

def add_gaussian_noise(img, mean=0, sigma=0.01,percentage=0.1):
    Noisy_img = img.copy()
    Noise_num = int (percentage*img.shape[0]*img.shape[1])
    for i in range(Noise_num):
        x = random.randint(0,img.shape[0]-1)
        y = random.randint(0,img.shape[1]-1)
        
        Noisy_img[x,y] = Noisy_img[x,y] + random.gauss(mean,sigma)
        
        if Noisy_img[x,y] < 0:
            Noisy_img[x,y] = 0
        elif Noisy_img[x,y] > 255:
            Noisy_img[x,y] = 255
    
    return Noisy_img

img1 = cv2.imread('lenna.png')
imggray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('lenna.png',0)
Noisyimg = add_gaussian_noise(img2,mean=2,sigma=4,percentage=0.8)
cv2.imshow('lenna',imggray)
cv2.imshow('lenna_GaussianNoise',Noisyimg)
cv2.waitKey(0)
