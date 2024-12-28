import cv2
import numpy as np
def avg_hash(img):
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 求均值
    s = 0
    for i in range(8):
        for j in range(8):
            s = s + gray[i,j]
    avg = s/64
    # 遍历，转化哈希值为字符串
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str
def cmpHash(hash1,hash2):
    n = 0
    if len(hash1) != len(hash2):
        return 'Hash长度不等，报错'
    for i in range(len(hash1)) :
        if hash1[i] != hash2[i]:
            n +=1
    return n
img1 = cv2.imread('../../../request/task2/lenna.png')
img2 = cv2.imread('../../../request/task2/lenna_noisy.png')
hash1 = avg_hash(img1)
hash2 = avg_hash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1,hash2)
print('均值哈希算法中汉明距离：',n)
