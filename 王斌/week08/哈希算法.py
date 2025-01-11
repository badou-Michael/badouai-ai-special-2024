import cv2
import numpy as np


def meanHash(img):
    img = cv2.resize(img,(8,8),interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgMean = np.mean(gray);
    hash_str = ''
    for i in range(8):
        for j in range(8):
           if gray[i,j]>imgMean:
               hash_str = hash_str + "1"
           else:
               hash_str = hash_str + "0"
    return hash_str

def differenceHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                hash_str = hash_str + "1"
            else:
                hash_str = hash_str + "0"
    return hash_str

def cmpHash(var1, var2):
    n = 0
    if len(var1) != len(var2):
        return -1
    for i in range(len(var1)):
        if var1[i] != var2[i]:
            n = n + 1
    return n

img1 = cv2.imread("C:/Users/Administrator/Desktop/123.jpg")
img2 = cv2.imread("C:/Users/Administrator/Desktop/234.jpg")

a = meanHash(img1)
b = meanHash(img2)
n=cmpHash(a,b)
print('均值哈希算法相似度：',n)

a = differenceHash(img1)
b = differenceHash(img2)
n=cmpHash(a,b)
print('差值哈希算法相似度：',n)
