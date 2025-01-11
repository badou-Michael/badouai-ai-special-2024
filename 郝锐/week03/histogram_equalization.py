import cv2
import numpy as np
from matplotlib import pyplot as plt
'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可,最后用一个变量来接收
'''
image_gray = cv2.imread("lenna.png",0)
#cv2.imshow("image_gray", image_gray)
#均衡化前的直方图
plt.subplot(411)
plt.title("gray_before_equalization")
hist = cv2.calcHist([image_gray],[0],None,[256],[0,256])
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)
plt.xlim([0,256])#设置x坐标轴范围
#灰度图直方图均衡化
dst_image = cv2.equalizeHist(image_gray)
# 均衡化后直方图
plt.subplot(412)
plt.title("gray_after_equalization")
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.xlim([0,256])#设置x坐标轴范围
hist = cv2.calcHist([dst_image],[0],None,[256],[0,256])
plt.hist(dst_image.ravel(), 256)

cv2.imshow("image_gray_equalization_before_after",np.hstack([image_gray,dst_image]))
cv2.waitKey(0)

#彩色图像的直方图均衡化
image_color = cv2.imread("lenna.png",1)
# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
b,g,r = cv2.split(image_color)
bEQ = cv2.equalizeHist(b)
gEQ = cv2.equalizeHist(g)
rEQ = cv2.equalizeHist(r)
#合并通道
dst_color = cv2.merge((bEQ,gEQ,rEQ))
#原始彩色图像的直方图
plt.subplot(413)
channels = cv2.split(image_color)
colors = ("b","g","r")
plt.title("cv2-calcHist-colored-before")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for channel,color in zip(channels,colors):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
#均衡化后的图像直方图
plt.subplot(414)
channels = cv2.split(dst_color)
colors = ("b","g","r")
plt.title("cv2-calcHist-colored-after")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for channel,color in zip(channels,colors):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.show()

cv2.imshow("image_color_equalization_before_after",np.hstack([image_color,dst_color]))
cv2.waitKey(0)
