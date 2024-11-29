import cv2
import numpy as np
 
#均值哈希算法 average
def aHash(img):
    #缩放为8*8灰度图，注意是（宽度*高度）=（列数*行数），interpolation是插值法（最近邻插值法/双线性插值法/三次插值法）
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    s=0
    hash_str=''
    
    #求平均灰度
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    avg=s/64
    
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'            
    return hash_str
 
#差值算法 differ
def dHash(img):
    #缩放8*9灰度图
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    hash_str=''
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str
 
#Hash值对比
def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断,不相等则n计数+1，n最终为相似度
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n=n+1
    return n
 
img1=cv2.imread('lenna.png')
img2=cv2.imread('lenna_noise.png')
hash1= aHash(img1)
hash2= aHash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('均值哈希算法相似度：',n)
 
hash1= dHash(img1)
hash2= dHash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('差值哈希算法相似度：',n)
