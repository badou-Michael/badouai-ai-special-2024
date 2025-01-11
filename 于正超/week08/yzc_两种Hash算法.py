import cv2
from skimage import util
import numpy as np

"""
利用函数接口，生成噪声图像
"""
img = cv2.imread("..\\lenna.png")
img_nosiy = util.random_noise(img,'gaussian')
img_nosiy_unit8 = (img_nosiy * 255).astype(np.uint8)
# cv2.imwrite("img_nosiy.png",img_nosiy_unit8)
# cv2.imshow("nosiy",img_nosiy)
# cv2.waitKey()

"""
均值哈希
1.缩放（8*8）  2.转灰度   减小计算量
3.求均值   4.比较  5.生成hash值  6.对比
"""
def aHash(img):
    img = cv2.resize(img,(8,8))
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    avg = np.mean(img_gray)
    print(avg)
    print("__________________________")
    # s = 0
    # for i in range(8):
    #     for j in range(8):
    #         s += img_gray[i,j]
    # avg1 = s / 64
    # print(avg1)

    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i,j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str +'0'
    return hash_str
"""
差值hash算法
1.缩放、灰度化
2.比较，前一个点与后一个点像素值大小
3.生成hash  4.对比
"""
def dHash(img):
    img = cv2.resize(img,(9,8))
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i,j] > img_gray[i,j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return  hash_str

"""
汉明距离计算两个hash差异
"""

def cmpHash(hash1,hash2):
    n = 0
    #防呆检查，hash长度不同，则返回-1报错
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n


if __name__ == '__main__':
    hash1 = aHash(img)  ##均值哈希
    hash2 = aHash(img_nosiy_unit8)
    n = cmpHash(hash1,hash2)
    print("均值哈希算法相似度：",n)
    hash3 = dHash(img)  ##插值哈希
    hash4 = dHash(img_nosiy_unit8)
    n = cmpHash(hash3,hash4)
    print("插值哈希算法相似度：",n)
