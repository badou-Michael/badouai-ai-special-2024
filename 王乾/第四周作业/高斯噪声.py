import numpy as NP
import cv2
from numpy import shape
import random

#创建高斯噪声的函数
def GaussianNoise(image,means,sigma,percetage):
    '''
    实现对图片添加高斯噪声
    :param image: 输入图片
    :param means: 正态分布的中心位置
    :param sigma:标准差
    :param percetage:处理图像的百分比
    :return:处理后的图片
    '''
    NoiseImg = image
    NoiseNum = int(percetage*image.shape[0]*image.shape[1])        #计算需要参与计算的像素数量
    for i in range(NoiseNum):
        #每次取随机一个点
        #用random.randint生成随机整数，来生成像素的X,Y位置
        pointX = random.randint(0,image.shape[0]-1)
        pointY = random.randint(0,image.shape[1]-1)
        #在原有的像素值上添加高斯随机数
        NoiseImg[pointX,pointY] = NoiseImg[pointX,pointY]+random.gauss(means,sigma)
        #判断是否超过阈值，使其不小于0，不大于255
        if NoiseImg[pointX,pointY] < 0:
            NoiseImg[pointX, pointY] = 0
        elif NoiseImg[pointX,pointY] > 255:
            NoiseImg[pointX,pointY] = 255
    return NoiseImg

image = cv2.imread("lenna.png",0)
image1 = GaussianNoise(image,2,4,0.8)
image = cv2.imread("lenna.png")
image2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("窗口1",image1)
cv2.imshow("窗口2",image2)
cv2.waitKey()
