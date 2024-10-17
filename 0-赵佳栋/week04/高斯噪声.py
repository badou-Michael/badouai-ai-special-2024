'''
@Project ：BadouCV 
@File    ：gauss_noise.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/14 23:59 
'''
import random

import cv2


# 手动实现高斯噪声
def to_gaussNoise (src_img,mean,sigma, noiseLevel):
    '''

    :param src_img: 原图
    :param mean: mean（期望值）
    :param sigma: sigma（标准差）
    :param noiseLevel: 噪声等级
    :return: img
    '''

    img = src_img
    h,w = img.shape[:2]
    noiseNum = int(noiseLevel * h * w)  # 噪声像素个数

    # 给像素点赋值高斯随机数
    for i in range(noiseNum):
        x = random.randint(0,w-1)
        y = random.randint(0,h-1)
        img[x,y] = img[x,y] + random.gauss(mean,sigma)

        # 传进来的图片是单通道的就直接判断，如果是一张三通道图片 img[x, y]代表有三个值，就要加any
        if  img[x, y].any()<0:
            img[x, y]=0
        elif img[x, y].any()> 255:
            img[x, y]=255

    return img


img=cv2.imread('../lenna.png')
noise_img=to_gaussNoise(img,8,8,0.8)
noise_img_gray=to_gaussNoise(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),2,6,0.8)

cv2.imshow('src',img)
cv2.imshow('noise',noise_img)
cv2.imshow('noise2',noise_img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
