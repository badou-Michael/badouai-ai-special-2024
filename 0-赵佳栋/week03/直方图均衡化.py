'''
@Project ：BadouCV 
@File    ：img_hist_Equalize.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/08 21:03
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('../lenna.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 通过均衡化函数把灰度图转化成均衡化后的图out_img
out_Grayimg = cv2.equalizeHist(gray)
# 绘制均衡化后灰度图的直方图
out_Grayimg_hist=cv2.calcHist([out_Grayimg],[0],None,[256],[0,256])
src_Grayimg_hist=cv2.calcHist([gray],[0],None,[256],[0,256])

plt.figure()
plt.title('out_Grayimg_equalizeHist')
plt.xlim([0,256])
plt.xlabel('bins')
plt.ylabel('pixels')
plt.plot(out_Grayimg_hist)
plt.plot(src_Grayimg_hist,color='r')
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([gray, out_Grayimg]))
# cv2.waitKey(0)


# 彩色 三通道 直方图均衡化

#分割三通道
chanels = cv2.split(img)
# print(type(chanels))
#chanels是个元组，包含每个通道的像素数据，以及数据类型
# 分别对三个通道直方图均衡化
out_b = cv2.equalizeHist(chanels[0])
out_g = cv2.equalizeHist(chanels[1])
out_r = cv2.equalizeHist(chanels[2])

# 合并三通道
out_img = cv2.merge((out_b,out_g,out_r))

cv2.imshow('out_img_equalizeHist',np.hstack([img, out_img]))
# cv2.waitKey(0)


plt.figure()
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("Pixels")

for (chan,color) in zip((out_b,out_g,out_r),('b','g','r')):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color=color)
    plt.xlim([0,256])
plt.show()

cv2.waitKey(0)
