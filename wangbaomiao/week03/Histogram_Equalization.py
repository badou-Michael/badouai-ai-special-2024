# -*- coding: utf-8 -*-
# time: 2024/10/17 12:35
# file: Histogram_Equalization.py
# author: flame
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取彩色图像
img = cv2.imread("lenna.png",1)
# 将彩色图像转换为灰度图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 显示灰度图像
#cv2.imshow("image_gray",gray)
# 等待用户按键，暂停程序执行
#cv2.waitKey()

# 对灰度图像进行直方图均衡化
dst = cv2.equalizeHist(gray)
# 计算均衡化后图像的直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
# 绘制直方图
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

# 显示原始灰度图像和直方图均衡化后的图像
cv2.imshow("Histogram_Equalization",np.hstack([gray,dst]))
# 等待用户按键，暂停程序执行
cv2.waitKey(0)

# 读取图像文件，保持原始色彩
img = cv2.imread("lenna.png",1)
# 显示原始图像
cv2.imshow("src",img)
# 等待用户按键，暂停程序执行
cv2.waitKey()

# 分离图像的BGR通道
(b,g,r) = cv2.split(img)
# 对每个通道进行直方图均衡化，以提升对比度和亮度
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并处理后的通道，形成均衡化后的图像
result = cv2.merge((bH,gH,rH))
# 显示均衡化后的图像
cv2.imshow("dst_rgb",result)
# 等待用户按键，暂停程序执行
cv2.waitKey()

# 并排显示原始图像和均衡化后的图像，便于对比效果
cv2.imshow("his_dst_rgb",np.hstack([img,result]))
# 等待用户按键，暂停程序执行
cv2.waitKey()
