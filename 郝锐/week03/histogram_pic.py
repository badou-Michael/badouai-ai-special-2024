import cv2
import numpy as np
from matplotlib import pyplot as plt
'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''
#灰度图的直方图
#先获取灰度图
#imread("lenna.png",flag=-1,0,1)
#-1 表示读取原始图像，包括alpha通道（如果存在）
#0 表示以灰度模式读取图像。
#1 表示读取彩色图像（三通道）。
#img = cv2.imread("lenna.png",0) 直接读取灰度图
img = cv2.imread("lenna.png",1)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(type(img_gray))
#print(img_gray)
cv2.imshow("image_grey",img_gray)
#cv2.waitKey(0)
#灰度图的直方图
#新建一个图像
plt.subplot(311)
#plt.hist(data, bins=None, range=None, density=None, ...)
#bins 参数用于指定数据分成的区间数量，range 参数用于指定数据的取值范围
# #ravel()可以加速计算直方图。将多为数组转换为一维数组
plt.title("plt.hist")
plt.hist(img_gray.ravel(),256)
#plt.show()
#灰度直方图方法二
plt.subplot(312)
hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
plt.title("cv2-calcHist-gray")
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)
plt.xlim([0,256])#设置x坐标轴范围
#plt.show()

#彩色图像的直方图
img_colored = cv2.imread("lenna.png")
#cv2.imshow("colored image",img_colored)
#cv2.waitKey(0)
#取的图像分解成单独的颜色通道。
channels = cv2.split(img_colored)
colors = ("b","g","r")
plt.subplot(313)
plt.title("cv2-calcHist-colored")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
#zip 内置函数，将多个可迭代对象压缩成一个可迭代元组对象
for channel,color in zip(channels,colors):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()
