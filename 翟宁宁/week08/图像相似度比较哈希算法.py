'''
图像相似度比较哈希算法

均值哈希算法
步骤
1. 缩放：图片缩放为8*8，保留结构，除去细节。
2. 灰度化：转换为灰度图。
3. 求平均值：计算灰度图所有像素的平均值。
4. 比较：像素值大于平均值记作1，相反记作0，总共64位。
5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不相同位数越少，图片越相似。
描述信息：这里缩放8*8的目的保留结构，去除细节，同时减少计算量。
假设一张图片是1080*1080，按照步骤，我们要在100多万个数的求和、比较 ，完全不是一个量级的。
'''

import cv2


'''
均值哈希算法
input:img
output:str  
'''
def avg_hash(img):
    # 图像缩放，下采样.差值使用双线性插值
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_LINEAR)
    # gray图
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #avg均值
    avg = img_gray.mean()

    #4. 比较：像素值大于平均值记作1，相反记作0，总共64位。
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i,j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str

'''
插值哈希算法
步骤
1. 缩放：图片缩放为8*9，保留结构，除去细节。
2. 灰度化：转换为灰度图。
3. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，
八个差值，有8行，总共64位
4. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
7. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样
的，不相同位数越少，图片越相似。
'''
def inter_hash(img):
    # 图像缩放，下采样.差值使用双线性插值 h,w
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_LINEAR)
    # gray图
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #avg均值
    #avg = img_gray.mean()

    #4. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i,j] > img_gray[i,j+1]:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str


def cmphash(hash1,hash2):
    #记录二进制对应位置不同的个数
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n+1

    return n



img1 = cv2.imread('../images/lenna.png')
img2 = cv2.imread('../images/pop.png')
hash1 = avg_hash(img1)
hash2 = avg_hash(img2)
print(hash1)
print(hash2)
n1=  cmphash(hash1,hash2)
print('img1和img2的均值相似度',n1)
print('======================')
hash3 = inter_hash(img1)
hash4 = inter_hash(img2)
n2 = cmphash(hash3,hash4)
print(hash3)
print(hash4)
print('img1和img2的插值相似度',n2)

