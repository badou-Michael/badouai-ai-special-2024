import cv2
import numpy as np

# 均值哈希
def avgHash(img):
    # 图片缩放成8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)

    # 转换成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0;
    hash_str = ''

    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]

    # 求平均灰度值
    avg = s / 64

    # 灰度值大于avg，设为1，否则设为0,生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'

    return hash_str

# 差值哈希
def dHash(img):
    # 图片缩放成8*9
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)

    # 转换成灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 哈希初值为0
    hash_str = ''

    # 遍历像素并求和,每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str
# hash值比对
def comHash(hash1,hash2):
    count = 0

    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1

    # 不相等则count计数+1，count最终为相似度
    for i in range(len(hash1)):
        count = count + 1
    return count

img1 = cv2.imread('F:\DeepLearning\Code_test\lenna.png')
img2 = cv2.imread('F:\DeepLearning\Code_test\salt.png')
hash1 = avgHash(img1)
hash2 = avgHash(img2)
print('lenna哈希均值为：',hash1)
print('salt哈希均值为：',hash2)
count = comHash(hash1,hash2)
print('均值哈希算法相似度：',count)



hash1 = dHash(img1)
hash2 = dHash(img2)
print('lenna哈希差值为：',hash1)
print('salt哈希差值为：',hash2)
count = comHash(hash1,hash2)
print('差值哈希算法相似度：',count)

