import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from skimage import util
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 1: 高斯噪声
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

# 2.椒盐噪声
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


# 3: 噪声接口调用
img = cv2.imread('lenna.png', 1)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img1 = util.random_noise(img, mode='poisson')
img1 = util.random_noise(img, mode='gaussian')
cv2.imshow('source', img)
cv2.imshow('noiseimage', img1)
cv2.waitKey(0)
