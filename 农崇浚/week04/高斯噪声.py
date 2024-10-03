import cv2
import random

def GaussNoise(src, means, sigma):
    NoiseImg = src
    #NoiseNum = int(pencetage*src.shape[0]*src.shape[1])

    for x in range(int(src.shape[0])):
        for y in range(int(src.shape[1])):
            NoiseImg[x, y] = NoiseImg[x, y] + random.gauss(means,sigma)
            if NoiseImg[x, y] < 0:
                NoiseImg[x, y] = 0
            elif NoiseImg[x, y] > 255:
                NoiseImg[x, y] = 255
    return NoiseImg

img = cv2.imread('lenna.png',0)
img1 = GaussNoise(img,9,10)
cv2.imshow('lennao_gaussNoise',img1)

img2 = cv2.imread('lenna.png',0)
img3 = cv2.cvtColor(img2,cv2.COLOR_BGR2BGRA)
cv2.imshow('lenna',img3)

cv2.waitKey(0)
