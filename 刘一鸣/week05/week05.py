'''
第五周作业：实现canny（尽量用手写实现）
1、高斯平滑：通过高斯滤波器平滑图像，去除噪声。
2、计算梯度：使用Sobel算子计算图像的梯度强度和方向。
3、非极大值抑制：保留局部最大值，去除不属于边缘的点，细化边缘。
4、双阈值检测：通过高、低阈值将像素分类为强边缘、弱边缘和非边缘。
5、边缘连接：通过边缘跟踪，将弱边缘与强边缘连接，形成完整的边缘图像。
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('lenna.png',1)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_res=cv2.Canny(img,100,200)
cv2.imshow('img',img_res)
cv2.waitKey(0)
cv2.destroyAllWindows()
