import cv2
import random
from numpy import shape

def PepperSalt(src,percentage):
    NoiseImg = src
    NoiseNum = int(src.shape[0]*src.shape[1]*percentage)
    # 随机取一个像素点
    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)

        if random.random() < 0.5:   # random.random生成随机浮点数,取到的像素值随机变成黑点0和白点255
            NoiseImg[randX,randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg

img = cv2.imread('F:\DeepLearning\Code_test\lenna.png',0)
img1 = PepperSalt(img,0.6)
#cv2.imwrite('PepperSalt',img1)

img = cv2.imread('F:\DeepLearning\Code_test\lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_lenna',img2)

cv2.imshow('PepperSalt_lenna',img1)
cv2.waitKey(0)
