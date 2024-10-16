import cv2
import numpy as np
from matplotlib import pyplot as plt


# 灰度图像直方图
# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image_gray", gray)

# 灰度图像的直方图，方法一
plt.figure()
plt.hist(gray.ravel(), 256)
plt.show()

'''
# 灰度图像的直方图, 方法二
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()#新建一个图像
plt.title("Grayscale Histogram")
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)
plt.xlim([0,256])#设置x坐标轴范围
plt.show()
'''

'''
#彩色图像直方图

image = cv2.imread("lenna.png")
cv2.imshow("Original",image)
#cv2.waitKey(0)

chans = cv2.split(image)
colors = ("b","g","r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])
plt.show()
'''

