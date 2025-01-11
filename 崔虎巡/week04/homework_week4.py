import numpy as np
import random
import cv2
from numpy import shape
def GaussianNoise(src,means,sigma,percentage):
    noiseimg = src
    noisenum = int(src.shape[0] * src.shape[1] * percentage)
    for i in range(noisenum):
        randY = random.randint(0,src.shape[0] - 1)
        randX = random.randint(0,src.shape[1] - 1)
        noiseimg[randY,randX] = noiseimg[randY,randX] + random.gauss(means,sigma)
        if noiseimg[randY,randX] < 0:
            noiseimg[randY,randX] = 0
        elif noiseimg[randY,randX] > 255:
            noiseimg[randY,randX] = 255
    return noiseimg
def saltpepper_noise(src,per):
    noiseimg = src
    noisenum = int(src.shape[0]*src.shape[1]*per)
    for i in range(noisenum):
        randY = random.randint(0,src.shape[0] - 1)
        randX = random.randint(0,src.shape[1] - 1)

        if random.random() <= 0.5:
            noiseimg[randY,randX] = 0
        else:
            noiseimg[randY,randX] = 255
    return noiseimg
class CPCA(object):
    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.centrX = []
        self.C = []    #样本集的协方差矩阵
        self.U = []     #样本集的降维转换矩阵
        self.Z = []     #样本集的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        return centrX

    def _cov(self):
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        return C

    def _U(self):

        a, b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        ind = np.argsort(-1 * a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z



if __name__ == '__main__':
    salt_pepper_enable = 0
    gaussian_noise_enable = 0
    PCA_ENABLE = 1
    if salt_pepper_enable == 1:
        img =cv2.imread('lenna.png',0)
        #cv2.imshow('source', img)
        noiseimg = saltpepper_noise(img,0.8)
        img = cv2.imread('lenna.png')
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('source', img2)
        cv2.imshow('pepper&salt_noise',noiseimg)
        cv2.waitKey(0)
    if gaussian_noise_enable == 1:
        img = cv2.imread('lenna.png',0)
        noiseimg = GaussianNoise(img,2,4,0.8)
        img = cv2.imread('lenna.png')
        img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('source',img2)
        cv2.imshow('gaussian_noise',noiseimg)
        cv2.waitKey(0)

    if PCA_ENABLE:
        X =np.array([[10, 15, 29],
                      [15, 46, 13],
                      [23, 21, 30],
                      [11, 9,  35],
                      [42, 45, 11],
                      [9,  48, 5],
                      [11, 21, 14],
                      [8,  5,  15],
                      [11, 12, 21],
                      [21, 20, 25]])
        K = np.shape(X)[1] - 1
        pca = CPCA(X, K)
