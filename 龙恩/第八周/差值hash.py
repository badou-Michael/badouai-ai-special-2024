import cv2
import numpy as np

def bhash(img):
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hashstr=''
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                hashstr+="1"
            else:
                hashstr+='0'
    return hashstr

def com(hash1,hash2):
    n=0

    if len(hash1)!=len(hash2):
        return -1
    for i in range(len(hash1)):

        if hash1[i]!=hash2[i]:
            n=n+1
    return n

img1=cv2.imread('lenna.png')
img2=cv2.imread('lenna_noise.png')
hash1= bhash(img1)
hash2= bhash(img2)
print(hash1)
print(hash2)
n=com(hash1,hash2)
print('均值哈希相似度：',n)
