#差值哈希算法

import cv2
import numpy as np

#计算哈希值
def cha_hash (img):
    #将图片缩放为9*8,interpolation用于指定插值方法
    img = cv2.resize(img,(9,8),interpolation = cv2.INTER_AREA)
    #将图片灰度化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #前一个像素点与后一个像数值点进行比较，大于记作1，小于记作0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str
#哈希值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

img1=cv2.imread('lenna.png')
img2=cv2.imread('lenna04_1.png')
hash1= cha_hash(img1)
hash2= cha_hash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('差值哈希算法相似度：',n)
