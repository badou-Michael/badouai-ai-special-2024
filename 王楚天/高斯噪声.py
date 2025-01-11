import cv2
import numpy as np
import random
def gauss(img,means,sigma,per):
  noisyimg=img
  num=int(per*noisyimg.shape[0]*noisyimg.shape[1])
  for i in range(num):
    randx=random.randint(0,noisyimg.shape[0]-1)
    randy=random.randint(0,noisyimg.shape[1]-1)
    noisyimg[randx,randy]+=random.gauss(means,sigma)
    if noisyimg[randx,randy]<0:
      noisyimg[randx,randy]=0
    if noisyimg[randx,randy]>255:
      noisyimg[randx,randy]=255
  return noisyimg
img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)
      
