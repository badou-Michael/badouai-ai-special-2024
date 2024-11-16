import cv2
import numpy as np


# 均值哈希算法
def aHash(img):

    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s 像素值和，初始设为0
    s = 0
    #  hash_str 转换为二进制的字符串，初始设为“”
    hash_str = ''
    # 求此图像的像素均值
    for i in range(8):
        for j in range(8):
            s = s+gray[i,j]
    avg = s/64
    # 求二进制字符串（hash值），像素大于均值的为1，小于均值为0
    for i in range(8):
        for j in range(8):
            if gray[i,j]>avg:
                hash_str  = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return  hash_str
    # 均值哈希算法

# 差值哈希算法
def dHash(img):
    # 缩放为8*9，resize(wight,height)
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s 像素值和，初始设为0
    s = 0
    #  hash_str 转换为二进制的字符串，初始设为“”
    hash_str = ''

    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

#汉明距离，Hash值对比
def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断，每个位置是否一致，不一致的计算加1
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n


img1 = cv2.imread('iphone1.png')
img2 = cv2.imread('iphone2.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', n)