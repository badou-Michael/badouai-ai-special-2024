import cv2
import numpy as np
from skimage import util

'''
图像相似度比较
'''


# 均值哈希算法
def aHash(img):
    # 缩放图像为 8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)  # cv2.INTER_CUBIC：三次插值（效果更好，适合缩小图像，但速度较慢）

    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 求平均灰度
    avg = np.mean(gray)

    # 初始化hash值
    hash_str = ''

    # 灰度大于平均值为1相反为0生成对应hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值哈希算法
def dHash(img):
    # 缩放图像为8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 初始化hash值
    hash_str = ''

    # 每行前一个像素大于后一个像素为1，反之为0，生成对应hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# hash值比较
def cmpHash(hash1, hash2):
    # 初始化
    n = 0

    # 如果hash值位数不相等则返回-1报错（防呆检查）
    if len(hash1) != len(hash2):
        return -1

    # 遍历判断
    for i in range(len(hash1)):
        # 对应位置不相等记为1
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


# 测试
img1 = cv2.imread('lenna.png')
img2 = util.random_noise(img1, mode='gaussian', mean=0.5, var=0.25)
img2 = (img2*255).astype('uint8') # openCV读图需要转成整型

hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1,hash2)
print('均值哈希算法图像相似度：',n)

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1,hash2)
print('差值哈希算法图像相似度：',n)
    
    
    
    
