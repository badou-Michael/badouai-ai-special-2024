import cv2
import numpy as np
from numpy import shape
import random

def pepper_noise(img, percentage):
    Noiseimg = img
    Noisenum = int(percentage * img.shape[0] * img.shape[1])
    for i in range(Noisenum):
        randX = random.randint(0, img.shape[0]-1)
        randY = random.randint(0, img.shape[1]-1)
        
        if random.random() <= 0.5:
            Noiseimg[randX, randY] = 0
        else:
            Noiseimg[randX,randY] = 255
    return Noiseimg

path = "/Users/zhouzhaoyu/Desktop/ai/lenna.png"
img = cv2.imread(path,0)
img1 = pepper_noise(img, 0.8)
cv2.imshow("pepper", img1)
cv2.waitKey(0)
