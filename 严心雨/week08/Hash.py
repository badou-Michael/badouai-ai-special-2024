import cv2
import numpy as np

#均值哈希算法
def aHash(image):

    """
    cv2.resize(data,dsize=,fx=,fy=,interpolation=)：通过插值的方式来改变图像的尺寸
    dsize: 代表期望的输出图像大小尺寸。dsize形参的数组的宽度在前，高度在后，所以当形参为(256,512)时，
    实际上得到的其实时512*256的图像
    fx: 代表水平方向上（图像宽度）的缩放系数
    fy: 代表竖直方向上（图像高度）的缩放系数，另外，如果dsize被设置为0(None),
    则按fx与fy与原始图像大小相乘得到输出图像尺寸大小
    interpolation:插值方式，默认选择线性插值，越负载的插值方式带来的改变和差异越大
                  INTER_NEAREST(最邻近插值)
                  INTER_CUBIC(三次样条插值)
                  INTER_LINEAR(线性插值)
                  INTER_AREA(区域插值)
    """
    # 将图片缩放为8*8，保留结构，除去细节
    image = cv2.resize(image,dsize=(8,8),interpolation = cv2.INTER_CUBIC)
    #转为灰度图
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #求像素平均值
    #a 求像素SUM值
    s = 0
    aHash = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i,j]
    #b 求像素平均值
    avg = s / 64

    #像素值比较：像素值大于平均值记为1，相反记作0，总共64位
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                Hash_str = aHash + '1'
            else:
                Hash_str = aHash + '0'
    return Hash_str

#差值哈希算法
def dHash(image):

    # 将图片缩放为8*9，保留结构，除去细节
    image = cv2.resize(image,dsize=(9,8),interpolation = cv2.INTER_CUBIC)
    #灰度化
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #比较：像素值大于后一个像素值记作1，相反记作0。
    bHash = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                Hash_str = bHash + '1'
            else:
                Hash_str = bHash + '0'
    return Hash_str

#Hash值对比
def compare(hash1,hash2):
    n = 0
    #防呆检查
    if len(hash1) != len(hash2):
        return -1
    #遍历检查
    for i in range(len(hash1)):
        for j in range(len(hash2)):
            hash1[i] != hash2[j]
            n = n + 1
    return n

#均值哈希算法两张图的哈希值（二进制）对比
image1 = cv2.imread('lenna.png')
image2 = cv2.imread('lenna_gaussi_noise.png')
hash1 = aHash(image1)
hash2 = aHash(image2)
print(hash1)
print(hash2)
n = compare(hash1,hash2)
print('均值哈希算法相似度：',n)

#差值哈希算法两张图的哈希值（二进制）对比
image1 = cv2.imread('lenna.png')
image2 = cv2.imread('lenna_gaussi_noise.png')
hash1 = dHash(image1)
hash2 = dHash(image2)
n = compare(hash1,hash2)
print('差值哈希算法相似度：',n)





