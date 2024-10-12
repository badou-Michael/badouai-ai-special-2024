# 本代码实现对图片添加噪声，分别实现添加高斯噪声和椒盐噪声
import random
import cv2

def noise_Gaussian(srcImg,mean,variance,percentage):
    noiseImg = srcImg.copy()
    #注意此处采用深拷贝，这是源于python中的赋值操作，是直接指向一块内存地址的，如果直接赋值，后续对noiseImg的修改也会修改srcImg本身
    noise_nums = int(percentage * noiseImg.shape[0] * noiseImg.shape[1])
    for i in range(noise_nums):
        randX = random.randint(0,noiseImg.shape[0]-1)
        randY = random.randint(0,noiseImg.shape[1]-1)
        noiseImg[randX,randY] = srcImg[randX,randY] + random.gauss(mean,variance)
        # 这里是标准差
        noiseImg[randX,randY] = 255 if noiseImg[randX,randY] > 255 else noiseImg[randX,randY]
        noiseImg[randX, randY] = 0 if noiseImg[randX, randY] < 0 else noiseImg[randX, randY]
    return noiseImg

def noise_PepperSalt(srcImg,percentage):
    noiseImg = srcImg.copy()
    noise_nums = int(percentage * noiseImg.shape[0] * noiseImg.shape[1])
    for i in range(noise_nums):
        randX = random.randint(0,noiseImg.shape[0]-1)
        randY = random.randint(0,noiseImg.shape[1]-1)
        noiseImg[randX,randY] = 255 if random.random() > 0.5 else 0
    return noiseImg

img = cv2.imread('lenna.png',0)
img_gauss = noise_Gaussian(img,0,0.1,1)
img_PepperSalt = noise_PepperSalt(img,0.8)
cv2.imshow('img',img)
cv2.imshow('img_gauss',img_gauss)
cv2.imshow('img_PepperSalt',img_PepperSalt)
cv2.waitKey(0)
