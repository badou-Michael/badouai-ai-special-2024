#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/10/31 14:45
# @Author: Gift
# @File  : 透视变幻.py 
# @IDE   : PyCharm
import cv2
import numpy as np
# 读取图像
img = cv2.imread('lenna.png')
y,x = img.shape[:2]
print(y)
print(x)
print(img.shape)
# 检查是否图像读取成功
if img is None:
    print("图像读取失败,请检查文件路径是否正确")
    exit()
# 定义透视变换的四个角点,左上角、右上角、右下角、左下角
src_img = np.float32([[0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0], [0, 0]])
# 定义目标图像上的四个点这里将图片缩放成二分之一大小
dst_img = np.float32([[0, int(img.shape[0]/2)], [int(img.shape[1]/2), int(img.shape[0]/2)], [int(img.shape[1]/2), 0], [0, 0]])
print(src_img)
print(dst_img)
# 计算透视变换矩阵
matrix_array = cv2.getPerspectiveTransform(src_img, dst_img)
# 进行透视变换
transformed_img = cv2.warpPerspective(img, matrix_array, (int(img.shape[1]/2), int(img.shape[0]/2)))
"""
img: 输入图像
matrix_array: 透视变换矩阵
dst_size: 输出图像大小
"""
cv2.imshow("original image", img)
cv2.imshow("transformed image", transformed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
