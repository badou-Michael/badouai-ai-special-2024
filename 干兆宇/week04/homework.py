import numpy as np
import cv2
from numpy import shape
import random
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):

        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)

        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        if  NoiseImg[randX, randY]< 0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg
img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)
#高斯

#椒盐
import numpy as np
import cv2
from numpy import shape
import random
def  fun1(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
	    randX=random.randint(0,src.shape[0]-1)
	    randY=random.randint(0,src.shape[1]-1)
	    if random.random()<=0.5:
	    	NoiseImg[randX,randY]=0
	    else:
	    	NoiseImg[randX,randY]=255
    return NoiseImg

img=cv2.imread('lenna.png',0)
img1=fun1(img,0.8)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_PepperandSalt',img1)
cv2.waitKey(0)

#PCA
# -*- coding: utf-8 -*-
"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""

import numpy as np


class CPCA(object):


    def __init__(self, X, K):

        self.X = X
        self.K = K
        self.centrX = []
        self.C = []
        self.U = []
        self.Z = []

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):

        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):

        ns = np.shape(self.centrX)[0]

        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):

        a, b = np.linalg.eig(
            self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)

        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):

        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    X = np.array([[10, 20, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)

