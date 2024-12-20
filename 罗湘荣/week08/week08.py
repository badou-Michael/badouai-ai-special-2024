import cv2
import numpy as np
import random
from skimage import util
#均值Hash
def aHash(photo):
    photo=cv2.resize(photo,(8,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
    s=0
    hash_str=''
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    avg=s/64
    for i in range(8):
        for j in range(8):
            if gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

#差值Hash
def dHash(photo):
    photo=cv2.resize(photo,(9,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

#对比
def campHash(hash1,hash2):
    n=0
    if len(hash1)!=len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

#高斯噪声；

photo1=cv2.imread('ho.jpg')
photo2=util.random_noise(photo1,mode='localvar')#噪声图片
photo2 = (photo2 * 255).astype(np.uint8)
hash1=aHash(photo1)
hash2=aHash(photo2)
n1=campHash(hash1,hash2)
print('均值哈希算法相似度：',n1)

hash3=dHash(photo1)
hash4=dHash(photo2)
n2=campHash(hash3,hash4)
print('差值哈希算法相似度：',n2)
