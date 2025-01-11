'''
@Project ：BadouCV 
@File    ：img_hist.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/13 16:57 
'''
import cv2
import numpy as np

#import cv2
import matplotlib.pyplot as plt


img = cv2.imread("../lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#灰度图像的直方图，方法一
plt.figure() # plt.figure() 创建一个新的图形对象
plt.hist(gray.ravel(), 256)
plt.show()

# 灰度图像的直方图, 方法二
''' 用openCV 库获取直方图
cv2.calcHist: 这是 OpenCV 中用于计算直方图的函数。
[gray]: 表示要计算直方图的图像。这里是一个列表，里面包含了灰度图像。
[0]: 指定要计算直方图的通道。对于灰度图像，只有一个通道，即第0个通道。
None: 表示没有使用掩膜。掩膜可以用来指定图像中要计算直方图的区域。
[256]: 指定直方图的 bin 数。对于 8 位灰度图像，通常使用 256 个 bin。
[0, 256]: 指定像素值的范围，这里是从 0 到 255。
'''

hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()#用plt库新建一个图像
plt.title("Grayscale Histogram") # 图像名
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist) # plt.plot()函数用于在二维平面上绘制线性图
plt.xlim([0,256])#设置x坐标轴范围
plt.show()


#彩色图像直方图

image = cv2.imread("../lenna.png")
cv2.imshow("Original",image)
#cv2.waitKey(0)

chans = cv2.split(image)
colors = ("b","g","r")
plt.figure()
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])
plt.show()
