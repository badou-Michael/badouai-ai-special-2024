#实现椒盐噪声
import random
import cv2

#定义椒盐噪声函数
def PepperSalt(src,percentage):
    NoiseImg=src
    NoiseNum=int(percentage*NoiseImg.shape[0]*NoiseImg.shape[1])
    for i in range(NoiseNum):
        #随机生成添加噪声的像素点坐标
        randx = random.randint(0, src.shape[0] - 1)
        randy = random.randint(0, src.shape[1] - 1)
        #添加椒盐噪声
        if random.random()<0.5:
            NoiseImg[randx,randy]=0
        else:
            NoiseImg[randx,randy]=255
    return NoiseImg

img=cv2.imread('lena.png',0)
img1=PepperSalt(img,0.5)
img=cv2.imread('lena.png',0)
cv2.imshow('source',img)
cv2.imshow('PepperSalt',img1)
cv2.waitKey(0)
