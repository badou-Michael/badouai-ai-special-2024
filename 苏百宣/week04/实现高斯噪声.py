# 实现高斯噪声
# author：苏百宣
import cv2
import random
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg = src
    NoiseNum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randx = random.randint(0,src.shape[0]-1)
        randy = random.randint(0,src.shape[1]-1)
        NoiseImg[randx,randy]=NoiseImg[randx,randy]+random.gauss(0,sigma)
        if NoiseImg[randx,randy] < 0:
            NoiseImg[randx,randy] = 0
        elif NoiseImg[randx,randy] > 255:
            NoiseImg[randx,randy] = 255
    return NoiseImg
img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('原图',img)
cv2.imshow('高斯处理图',img1)
cv2.waitKey(0)




