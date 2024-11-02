import numpy as np
import cv2
import  random
from skimage import util
def GaussianNoise(src,means,sigma,percentage):
    NoiseImg=src
    NoiseNum=int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randx = random.randint(0,src.shape[0]-1)
        randy = random.randint(0,src.shape[1]-1)
        NoiseImg[randx,randy] += random.gauss(means,sigma)
        NoiseImg[randx, randy] = min(max(NoiseImg[randx, randy], 0), 255)
    return NoiseImg

def peppersalt(src,percentage):
    NoiseImg = src.copy()
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randx = random.randint(0, src.shape[0] - 1)
        randy = random.randint(0, src.shape[1] - 1)
        if random.random()<=0.5:
            NoiseImg[randx, randy] =0
        else: NoiseImg[randx, randy] =255
    return  NoiseImg

#噪声接口调用
img=cv2.imread("lenna.png",0)
imgtest=util.random_noise(img,"salt")
cv2.imshow('source',img)
cv2.imshow('imgtest',imgtest)
cv2.waitKey(0)


img=cv2.imread("lenna.png",0)
img1=GaussianNoise(img,8,4,0.6)
img2=peppersalt(img,0.4)

cv2.waitKey(0)
cv2.imshow("source",img)
cv2.imshow("gauss",img1)
cv2.waitKey(0)
cv2.imshow("pepper",img2)
