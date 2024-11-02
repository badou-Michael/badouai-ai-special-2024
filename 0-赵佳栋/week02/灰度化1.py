#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：test01.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/10 15:25 
'''
import numpy as np
import cv2

#=========Demo01   RGB->单通道 （灰度化)===============
#     演示三种方式：浮点方法、整数方法、平均值法

img=cv2.imread('../lenna.png')
#手动转化
h,w=img.shape[:2] #获取图片宽高像素

result_img=np.zeros([h,w],img.dtype)# 创建一个与原图尺寸相同，但通道数为 1 的 空数组，用于存储灰度图像。数组的数据类型为原图类型

for i in range(w):
    for j in range(h):
        point=img[i,j]
        # OpenCV 三通道BGR的读取顺序，浮点算法
        result_img[i,j]=int(point[0]*0.11+point[1]*0.59+point[2]*0.3)
        ## 整数方法
        #result_img[i, j] = int((point[0] * 11 + point[1] *59 + point[2] * 30)/100)
        ## 平均数法（精度损失严重）
        #result_img[i,j]=int((point[0]+point[1]+point[2])/3)

#point是最后一个像素点的三通道值，是一个三位数数组 ，result_img[i,j]是最后一个像素点转化后的单通道值
print(point,result_img[i,j])
print(result_img)
print('image show gray------:%s'%result_img)
print('===================================')
cv2.imshow('imag',result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
