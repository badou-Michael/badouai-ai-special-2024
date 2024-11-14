import numpy as np
import cv2

# 均值哈希
def avghash(img):
    a = cv2.resize(img, (8,8), interpolation=cv2.INTER_LINEAR)
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    sum = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            sum += a[i][j]
    avg = sum/64
    A = ''
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] > avg:
                A += '1'
            else:
                A += '0'
    return A


# 差值哈希
def chazhihash(img):
    # resize函数里，dsize的第一个参数指定的是输出图像的【宽】，第二个才是【高】 --> 【宽前高后】
    # --> 与img.shape正好相反，使用cv2.resize函数的dsize参数需要特别注意。
    a = cv2.resize(img, (9,8), interpolation=cv2.INTER_LINEAR)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    A = ''
    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            if a[i][j] > a[i][j+1]:
                A += '1'
            else:
                A += '0'
    return A


def bijiao(A1, A2):
    if len(A1)!=len(A2):
        return -1
    sum = 0
    for i in range(len(A1)):
        if A1[i]==A2[i]:
            sum += 1
    return sum/64


img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
A1 = avghash(img1)
A2 = avghash(img2)
A3 = chazhihash(img1)
A4 = chazhihash(img2)
result1 = bijiao(A1,A2)
result2 = bijiao(A3,A4)
print('均值哈希算法下的图片相似度：{:.2f}%'.format(result1*100))
print('差值哈希算法下的图片相似度：{:.2f}%'.format(result2*100))
