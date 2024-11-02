#!/usr/bin/env python
# encoding=gbk

'''
Canny边缘检测：优化的程序
'''
import cv2
import numpy as np

def CannyThreshold(lowThreshold):
#detected_edges = cv2.GaussianBlur(gray,(3,3),0) #高斯滤波
#cv2.GaussianBlur(src, ksize, sigmaX, sigmaY=None, borderType=None)
##0 就是告诉 OpenCV 自动计算 X 方向和 Y 方向的标准差，因为这里没有指定 Y 方向的标准差，所以它将与 X 方向相同。
 # 函数返回一个新的图像，该图像是输入图像经过高斯模糊处理后的结果。
# sigmaX：高斯核在 X 方向上的标准差。如果 sigmaX 为 0，那么基于 ksize 宽度，sigmaX 会被自动计算。
#sigmaY：（可选）高斯核在 Y 方向上的标准差。如果为 None，则 sigmaY 将等于 sigmaX。如果 sigmaY 为 0，那么基于 ksize 高度，sigmaY 会被自动计算。
    detected_edges = cv2.Canny(gray
                               , lowThreshold
                               , lowThreshold * ratio
                               ,apertureSize=kernel_size)
##使用cv2.Canny函数进行边缘检测。这个函数接受四个参数：(输入的灰度图像,低阈值，高阈值，Sobel算子的孔径大小，这里设置为3)
    dst=cv2.bitwise_and(img,img,mask = detected_edges)
    cv2.imshow('canny result',dst)
#mask：（可选）操作的掩码，是一个8位单通道图像。掩码图像中非零像素的位置将被操作，零像素的位置将被设置为零。

lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread('lenna.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('canny result')
#设置调节杠,
'''
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''
cv2.createTrackbar('Min threshold','canny result',lowThreshold, max_lowThreshold, CannyThreshold)

CannyThreshold(0)  # initialization
if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2  ESC 键（ASCII 码为 27）
    cv2.destroyAllWindows()