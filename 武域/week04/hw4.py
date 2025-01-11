import numpy as np
import cv2
import random
from skimage.color import rgb2gray
from skimage import util
#image read
lenna = cv2.imread('/Users/leowu/Desktop/badou/week04/lenna.png')
lenna_GRAY = rgb2gray(lenna)
lenna_GRAY = (lenna_GRAY * 255).astype(np.uint8)

def gaussianNoise(image, ratio, sigma, means):
    h, w = image.shape[:2]
    noiseImg = image.copy()
    num = int(h * w * ratio)
    while(num):
        i = random.randrange(0, h)
        j = random.randrange(0, w)
        noiseImg[i][j] = noiseImg[i][j] + random.gauss(means, sigma)
        if noiseImg[i][j] > 255:
            noiseImg[i][j] = 255
        if noiseImg[i][j] < 0:
            noiseImg[i][j] = 0
        num = num - 1
    return noiseImg
lenna_GN = gaussianNoise(lenna_GRAY, 0.8, 4, 2)
# cv2.imshow('gaussian lenna', lenna_GN)
# cv2.waitKey(0)

def peppersaltNoise (image, ratio):
    H, W = image.shape
    noiseImage = image.copy()
    num = int(H * W * ratio)
    while(num):
        i = random.randrange(0, H)
        j = random.randrange(0, W)
        noiseImage[i][j] = random.choice([0,255])
        num = num - 1
    return noiseImage
lenna_PS = peppersaltNoise(lenna_GRAY, 0.5)
# cv2.imshow('ps lenna', lenna_PS)
# cv2.waitKey(0)

# API
noise_img=util.random_noise(lenna,mode='poisson')
cv2.imshow('poi lenna', noise_img)
cv2.waitKey(0)

#PCA
class CPCA(object):
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centerX = []
        self.C = []
        self.U = []
        self.Z = []

        self.centerX = self._centeralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centeralized(self):
        mean = np.array([np.mean(i)] for i in self.X.T)
        centerX = self.X - mean
        return centerX
    
    def _cov(self):
        ns = np.shape(self.centerX)[0]
        C = np.dot(self.centerX.T, self.centerX) / (ns - 1)
        return C
    
    def _U(self):
        a, b = np.linalg.eig(self.C)
        ind = np.argsort(-1 * a)
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U
    
    def _Z(self):
        Z = np.dot(self.X, self.U)

X = np.array([[10, 15, 29],
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
pca = CPCA(X,K)
