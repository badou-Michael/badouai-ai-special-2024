import random

import cv2
import numpy as np
from skimage import util
from sklearn.decomposition import PCA

#高斯
def GaussianNoise(src,means,sigma,percetage):
    img = src
    w = src.shape[0]
    h = src.shape[1]
    num = int(percetage*w*h)
    # f.循环按百分比随机获取部分像素
    for i in range(num):
        randomX = random.randint(0,w-1)
        randomY = random.randint(0,h-1)
        # a.输入参数sigma和mean
        # b.生成高斯随机数
        # d.根据输入像素计算出输出像素
        img[randomX,randomY] = img[randomX,randomY] + random.gauss(means,sigma)
        # e.重新将像素值放缩在[0~ 255]之间
        img[randomX, randomY] = np.clip(img[randomX,randomY],0,255)
        # g.输出图像
    return img

#椒盐
def SPNoise(src,snr):
    img = src
    w = src.shape[0]
    h = src.shape[1]
    num = int(w*h*snr)
    for i in range(num):
        randomX = random.randint(0,w-1)
        randomY = random.randint(0,h-1)
    #random.random()：调用时不需要参数，返回的是一个在 [0.0, 1.0) 范围内的浮点数。
    if random.random() <= 0.5 :
        img[randomX,randomY] = 0
    else :
        img[randomX,randomY] = 255
    return img

img = cv2.imread("d:\\Users\ls008\Desktop\lenna.png",0)
imgGauss = GaussianNoise(img,0,6,0.8)
imgSP = SPNoise(img,0.8)

#接口调用
noise_gs_img=util.random_noise(img,mode='poisson')
cv2.imshow('source',img)
cv2.imshow('GaussianNoise',imgGauss)
cv2.imshow('SPNoise',imgSP)
cv2.imshow('funNoise',noise_gs_img)
cv2.waitKey(0)

#问题总结：
#传图片的时候没有灰度化，如果是彩色图片img[randomX, randomY]就是[B, G, R]数组，灰度图片才是一个值，所以在if对比大小时跟单值对比会报错

#鸢尾花
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
pca = PCA(n_components=2)
pca.fit(X)
newX=pca.fit_transform(X)
print(newX)
