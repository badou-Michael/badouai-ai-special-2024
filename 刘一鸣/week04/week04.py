'''
1.实现高斯噪声、椒盐噪声手写实现
2.噪声接口调用
3.实现pca
'''
import random
import cv2
from skimage import util
import numpy as np
from sklearn.decomposition import PCA

#1.1高斯噪声
def GaussianNoise(src,means,sigma,percentage):
    NoiseImg=src.copy()
    NoiseNum=int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randx = random.randint(0,src.shape[0]-1)
        randy = random.randint(0,src.shape[1]-1)
        NoiseImg[randx,randy] += random.gauss(means,sigma)
        NoiseImg[randx, randy] = min(max(NoiseImg[randx, randy], 0), 255)
    return NoiseImg

#1.2椒盐噪声
def JiaoYan(src,percentage):
    NoiseImg = src.copy()
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randx = random.randint(0, src.shape[0] - 1)
        randy = random.randint(0, src.shape[1] - 1)
        if random.random()<=0.5:
            NoiseImg[randx, randy] =0
        else: NoiseImg[randx, randy] =255
    return  NoiseImg

img_G=cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img_with_Gnoise=GaussianNoise(img_G,4,4,0.6)

cv2.imshow('source_G',img_G)
cv2.imshow('Gaussian Noise',img_with_Gnoise)
cv2.waitKey(0)

img_J=cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img_with_Jnoise=JiaoYan(img_J,0.6)

cv2.imshow('source_J',img_J)
cv2.imshow('JiaoYan Noise',img_with_Jnoise)
cv2.waitKey(0)

#2.1噪声接口调用
img_Other=cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img_with_Sklearn=util.random_noise(img_Other,'pepper')
cv2.imshow('img_Other',img_Other)
cv2.imshow('img_with_Sklearn',img_with_Sklearn)
cv2.waitKey(0)

#3.1实现pca
X=np.array([[-1,2,3,-1],[5,5,5,3],[120,2,3,210],[56,34,22,121],[99,88,77,66],[43,12,98,99]])
pca=PCA(n_components=2)
pca.fit(X)
print(pca.fit_transform(X))
