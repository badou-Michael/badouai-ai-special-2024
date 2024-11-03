import cv2
from numpy import shape
import random

def GaoSiNoise(src,means,sigma,percentage):
    NoiseImg = src;
    NoiseNum = int(src.shape[0]*src.shape[1]*percentage)

    for i in range(NoiseNum):
        # randX、randY是噪声图的行和列；
        # 图片边缘不处理，所以要-1；
        # 利用函数random.randint生成随机数，并且该函数不会生成重复的随机数
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)

        # 然后在图片灰度图的行列数值上加上随机数
        NoiseImg[randX,randY] = NoiseImg[randX,randY] + random.gauss(means,sigma)

        # 如果灰度值小于0，则强制其为0；如果大于255，则强制为255
        if NoiseImg[randX,randY] < 0 :
            NoiseImg[randX,randY] = 0
        elif NoiseImg[randX,randY] > 255:
            NoiseImg[randX,randY] =255
    return NoiseImg

img = cv2.imread('F:\DeepLearning\Code_test\lenna.png',0)   # 0——灰度图片   1——彩色图片
img1 = GaoSiNoise(img,0,0.5,0.6)
img = cv2.imread('F:\DeepLearning\Code_test\lenna.png')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('src',img)
print("img__")
cv2.imshow("GaosiImage",img1)
cv2.imshow('gray',img2)
cv2.waitKey(0)

