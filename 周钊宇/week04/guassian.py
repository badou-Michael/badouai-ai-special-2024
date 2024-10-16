import cv2
import numpy as np
import random

def Gussian_noise(img, percentage, mean, sigma):
    Noiseimg = img
    NoiseNum = int(percentage*img.shape[0]*img.shape[1])

    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0]-1)
        randY = random.randint(0, img.shape[1]-1)

        tmpNoise = img[randX, randY] + random.gauss(mean, sigma)
        
        if tmpNoise < 0:
            tmpNoise = 0
        elif tmpNoise >255:
            tmpNoise = 255
        
        Noiseimg[randX, randY] = tmpNoise
    
    return Noiseimg

path = "/Users/zhouzhaoyu/Desktop/ai/lenna.png"
img = cv2.imread(path,0)
noise = Gussian_noise(img, 0.8, 4,2)
cv2.imshow("Gaussian noise", noise)
cv2.waitKey(0)