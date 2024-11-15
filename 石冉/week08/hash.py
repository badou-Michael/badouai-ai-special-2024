import cv2
from skimage import util
import numpy as np

#均值哈希函数
def ahash(src):
    dst=cv2.resize(src,(8,8),interpolation=cv2.INTER_LINEAR)
    img=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    s=0
    for i in range(8):
        for j in range(8):
            s=s+img[i,j]
    average=s/64
    hashstr=''
    for i in range(8):
        for j in range(8):
            if img[i,j]>average:
                hashstr=hashstr+'1'
            else:
                hashstr=hashstr+'0'
    return hashstr

#差值哈希函数
def dhash(src):
    dst = cv2.resize(src, (9, 8), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    hashstr=''
    for i in range(8):
        for j in range(8):
            if img[i,j]>img[i,j+1]:
                hashstr=hashstr+'1'
            else:
                hashstr=hashstr+'0'
    return hashstr

#对比两个哈希值函数
def comhash(hash1,hash2):
    com=0
    if len(hash1)!=len(hash2):
        return "Error: Hashes are of different lengths."
    else:
        for i in range(len(hash1)):
            if hash1[i]!=hash2[i]:
                com=com+1
    return com


#调用均值哈希hanshu
src=cv2.imread('lena.png')
gauss_img=util.random_noise(src,mode='gaussian').astype(np.uint8)
img1=ahash(src)
img2=ahash(gauss_img)
result=comhash(img1,img2)
print('均值哈希算法相似度：',result)

#调用差值哈希hanshu
src=cv2.imread('lena.png')
gauss_img=util.random_noise(src,mode='gaussian').astype(np.uint8)
img1=dhash(src)
img2=dhash(gauss_img)
result=comhash(img1,img2)
print('差值哈希算法相似度：',result)
