import numpy as np
import cv2
from numpy import shape
import random


def get_gauss_noise_image(src, means, sigma, percetage):
    NoiseImage = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])#得到添加高斯噪声的图片的大小
    
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        NoiseImage[randX,randY] = NoiseImage[randX,randY] + random.gauss(means,sigma)

        if  NoiseImage[randX, randY] < 0:
            NoiseImage[randX, randY] = 0
        elif NoiseImage[randX, randY] > 255:
            NoiseImage[randX, randY] = 255

    return NoiseImage


def get_pepper_salt_image(src, percetage):
    NoiseImage = src    
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])    
    for i in range(NoiseNum): 
	
        randX = random.randint(0, src.shape[0] - 1)       
        randY = random.randint(0, src.shape[1] - 1) 

        if random.random() <= 0.5:           
            NoiseImage[randX, randY] = 0       
        else:            
            NoiseImage[randX, randY] = 255    
    return NoiseImage


img = cv2.imread("lenna.png", 0)
GaussNoiseImage = get_gauss_noise_image(img, 2, 4, 0.9)
cv2.imshow('lenna_GaussianNoise', GaussNoiseImage)

PepperSaltImage = get_pepper_salt_image(img, 0.9)
cv2.imshow('lenna_PepperSalt', PepperSaltImage)
cv2.waitKey(0)
