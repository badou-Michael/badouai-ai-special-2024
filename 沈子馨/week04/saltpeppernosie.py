import cv2
import numpy as np
from numpy import shape
import random
def Saltpeppernoise(src, percetage):
     noiseimg = src
     noisenum = int(src.shape[0]*src.shape[1]*percetage)
     for i in range(noisenum):
         randx = random.randint(0, src.shape[0]-1)
         randy = random.randint(0, src.shape[1]-1)
         if random.random() < 0.5:
             noiseimg[randx, randy] = 0
         else:
             noiseimg[randx, randy] =255
     return noiseimg

img = cv2.imread('lenna.png',0)
img1 = Saltpeppernoise(img, 0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('origin', img2)
cv2.imshow('saltpepper',img1)
cv2.waitKey(0)