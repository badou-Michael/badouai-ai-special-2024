import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from skimage import util
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 1.1: 高斯噪声（默背手打的）
def gaosinoise(img, sigma, mean, percentage):
    noisenum = int(percentage*img.shape[0]*img.shape[1])
    for i in range(noisenum):
        rand_x = np.random.randint(0, img.shape[0]-1)
        rand_y = np.random.randint(0, img.shape[1]-1)
        img[rand_x, rand_y] = img[rand_x, rand_y] + random.gauss(mean, sigma)
        if img[rand_x, rand_y] < 0:
            img[rand_x, rand_y] = 0
        elif img[rand_x, rand_y] > 255:
            img[rand_x, rand_y] = 255
    return img


img1 = cv2.imread('lenna.png', 1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('lenna.png', 0)
img2 = gaosinoise(img2, 2, 4, 0.8)
cv2.imshow('1', img1)
cv2.imshow('2', img2)
cv2.waitKey(0)

# 1.2.椒盐噪声（默背手打的）
def jiaoyannoise(src, percentage):
      img = src
      noisenum = int(percentage*img.shape[0]*img.shape[1])
      for i in range(noisenum):
          rand_x = np.random.randint(0, img.shape[0]-1)
          rand_y = np.random.randint(0, img.shape[1]-1)
          if np.random.random() <= 0.5:
              img[rand_x, rand_y] = 0
          else: img[rand_x, rand_y] = 255
      return img


img1 = cv2.imread('lenna.png', 1)
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2 = cv2.imread('lenna.png', 0)
img2 = jiaoyannoise(img2, 0.8)
cv2.imshow('1', img1)
cv2.imshow('2', img2)
cv2.waitKey(0)










# 2: 噪声接口调用（默背手打的）
img = cv2.imread('lenna.png', 1)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img1 = util.random_noise(img, mode='poisson')
img1 = util.random_noise(img, mode='gaussian')
cv2.imshow('source', img)
cv2.imshow('noiseimage', img1)
cv2.waitKey(0)









# 3.PCA实现

# 3.1 最最细节实现（默背手打的）
class PCA(object):
    def __init__(self, x, k):
        self.x = x
        self.k = k
        self.A = self.A()
        self.C = self.C()
        self.U = self.U()
        self.Z = self.Z()

    def A(self):
        a = [np.mean(i) for i in self.x.T]
        b = self.x - a
        return b

    def C(self):
        c = np.dot(self.A.T, self.A)/(self.A.shape[0]-1)
        print(c)
        return c

    def U(self):
        eigvalue, eigvector = np.linalg.eig(self.C)
        a = np.argsort(-1*eigvalue)
        b = [eigvector[:,a[i]] for i in range(k)]
        b = np.array(b).T
        return b

    def Z(self):
        a = np.dot(x, self.U)
        print(a)
        return a


if __name__=='__main__':
    x = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    k = 2
    pca = PCA(x, k)


# 3.2 numpy细节实现（默背手打的）
class PCA():
    def __init__(self,k):
        self.k = k

    def fit_transform(self,x):
        x = x - x.mean(axis=0)
        c = np.dot(x.T,x)/(x.shape[1]-1)
        m,n = np.linalg.eig(c)
        o = np.argsort(-m)
        z = n[:,o[:self.k]]
        return np.dot(x,z)


pca = PCA(k=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
result = pca.fit_transform(X)
print(result)


# 3.3 sklearn细节实现（默背手打的）
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
pca = PCA(n_components=2)
XNEW = pca.fit_transform(X)
print(XNEW)


# 3.4 pca实现鸢尾花降维（默背手打的）
x, y = load_iris(return_X_y=True)
pca = PCA(n_components=2)
newx = pca.fit_transform(x)

a1 = []
a2 = []
b1 = []
b2 = []
c1 = []
c2 = []
for i in range(len(newx)):
    if y[i] == 0:
        a1.append(newx[i][0])
        a2.append(newx[i][1])
    elif y[i] == 1:
        b1.append(newx[i][0])
        b2.append(newx[i][1])
    else:
        c1.append(newx[i][0])
        c2.append(newx[i][1])
plt.scatter(a1, a2)
plt.scatter(b1, b2)
plt.scatter(c1, c2)
plt.show()
