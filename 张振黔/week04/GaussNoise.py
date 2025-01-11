import cv2
import numpy as np

def add_Gauss_Noise(image,mean,sigma):
    noise = np.random.normal(mean, sigma, image.shape) #生成随机正态分布矩阵
    Noise_img = image + noise
    Noise_img = np.clip(Noise_img, 0, 255).astype(np.uint8)  # 确保像素值在0到255之间
    return Noise_img

img = cv2.imread('lenna.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
Noise_img=add_Gauss_Noise(img,0,25)
imgs=np.hstack([img,Noise_img])
cv2.imshow('COMPARE',imgs)
cv2.waitKey()
