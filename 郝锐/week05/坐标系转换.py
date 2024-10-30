#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/10/13 17:21
# @Author: Gift
# @File  : 坐标系的转换.py 
# @IDE   : PyCharm
import numpy as np
#世界坐标系转换到相机坐标系
X_world = np.array([1, 2, 3 , 1])
print(X_world)
#假设一个旋转矩阵R
R_rotate = np.array([[0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]])
#假设一个偏移量t
T_shift = np.array([4, 5, 6])
#计算相机坐标 欧式变化 y=x*R_rotate+ T_shift
#np.dot表示俩个矩阵的乘法
P_camera = np.dot(R_rotate,X_world[:3]) + T_shift
print(P_camera)
#再进行相机坐标到图像物理坐标的转换，三维转二维
#相机内参矩阵K
f_x = 10 #相机在x轴上的焦距
f_y = 10 #相机在y轴上的焦距
c_x = 0 #相机在x轴上的中心点 通常为0
c_y = 0 #相机在y轴上的中心点 通常为0
K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]])
#计算相机物理坐标
Ph_camera = np.dot(K,P_camera) / P_camera[2]
print(Ph_camera)
#相机物理坐标再转换成像素坐标
#一下可以理解成相机的物理感光元器件的大小
pixel_size_x = 1000 #图像在x轴上的像素大小
pixel_size_y = 1000 #图像在y轴上的像素大小
#齐次坐标#此处的500是一个特定相机如上生成的1000*1000的图像，那么光心的坐标就是一半
np_tmp = np.array([[1/pixel_size_x, 0, 500],
                  [0, 1/pixel_size_y, 500],
                  [0, 0, 1]])
#像素坐标
final = np.dot(np_tmp,Ph_camera)
print(final[:2])
