import cv2
import numpy as np

def AHA_mean(img):
    img2 = cv2.resize(img,(8,8))

    #转为灰度图
    img_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    #求平均值
    sum = 0
    h, w = img_gray.shape
    for i in range(h):
        for j in range(w):
            sum += img_gray[i,j]
    avg = sum // 64

    #比较
    #生成哈希
    haxi = ''
    for i in range(h):
        for j in range(w):
            if img_gray[i,j] > avg:
                haxi += '1'
            else:
                haxi += '0'
    return haxi

def Compere_AHA(haxi1, haxi2):
    if len(haxi1) != len(haxi2):
        return -1
    num = 0
    for i in range(len(haxi1)):
        if haxi1[i] != haxi2[i]:
            num += 1

    return num

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_noise = cv2.imread('lenna_noise.png')
    img_haxi = AHA_mean(img)
    img_haxi_noise = AHA_mean(img_noise)

    num = Compere_AHA(img_haxi,img_haxi_noise)

    print('不相似数为:',num)

