import cv2
import numpy as np
def differ_hash(img):
    # resize的参数fx为沿着宽度的方向缩放，所以最终为8*9的像素图
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素值大于后一个像素值设为1，本行不和下一行比较，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return  hash_str
# 比较汉明距离
def cmpHash(hash1,hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n+=1
    return n
img1 = cv2.imread('../../../request/task2/lenna.png')
img2 = cv2.imread('../../../request/task2/lenna_noisy.png')
hash1 = differ_hash(img1)
hash2 = differ_hash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1,hash2)
print('均值哈希算法的汉明距离为：',n)