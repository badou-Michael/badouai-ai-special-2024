import random
from skimage import util
import cv2 as cv


def gaussianNoise(srcImg, means, sigma, percentage):
    noiseImg = srcImg
    noiseNum = int(percentage*srcImg.shape[0]*srcImg.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, srcImg.shape[1] - 1)  # W
        randY = random.randint(0, srcImg.shape[0] - 1)  # H
        noiseImg[randX, randY] = noiseImg[randX, randY] + random.gauss(means, sigma)
        noiseImg[randX, randY] = 0 if noiseImg[randX, randY] < 0 else noiseImg[randX, randY]
        noiseImg[randX, randY] = 255 if noiseImg[randX, randY] > 255 else noiseImg[randX, randY]
    return noiseImg


def saltPepperNoise(srcImg, percentage):
    noiseImg = srcImg
    noiseNum = int(percentage*srcImg.shape[0]*srcImg.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, srcImg.shape[1] - 1)  # W
        randY = random.randint(0, srcImg.shape[0] - 1)  # H
        noiseImg[randX, randY] = 0 if random.random() <= 0.5 else 255
    return noiseImg


img = cv.imread('lenna.jpg', 0)

gaussianNoiseImg = gaussianNoise(img,2,4,0.9)
saltNoiseImg = saltPepperNoise(img,0.9)

# 调用接口实现图像加噪
noise_po_img = util.random_noise(img, mode='poisson')
noise_sp_img = util.random_noise(img, mode='s&p')  # salt pepper
noise_gs_img = util.random_noise(img, mode='gaussian', mean=2, var=0.5)  # gaussian


cv.imshow('srcImg', img)
cv.imshow('gaussianNoiseImg', img)
cv.imshow('saltNoiseImg ', img)
cv.waitKey(0)
cv.destroyAllWindows()

