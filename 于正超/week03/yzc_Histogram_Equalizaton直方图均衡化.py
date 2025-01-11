"""
Histogram Equalization 直方图均衡化
yzc
灰度图直方图均衡化
1.导图、灰度化
2.灰度图直方图均衡化
3.生成直方图
===============
彩图直方图均衡化
1.通道分割cv2.split
2.每个通道进行均衡化
3.三个通道合并
4，彩图直方图
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 灰度图直方图均衡化
img = cv2.imread("..\\lenna111.png")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_Eq = cv2.equalizeHist(img_gray)

# 方法一
plt.figure()
plt.hist(img_Eq.ravel(),256)
plt.show()
# 方法二
# hist_img = cv2.calcHist([img_Eq],[0],None,[256],[0,256])
# plt.figure()
# plt.xlabel("Bins")
# plt.ylabel("number")
# plt.title("yzc gray Equalizaton_Hist")
# plt.xlim([0,256])
# plt.plot(hist_img)
# plt.show()

cv2.imshow("yzc gray hist:",np.hstack([img_gray,img_Eq]))
# cv2.waitKey()

# ==============分割线====================== #
# 彩图直方图均衡化
(b,g,r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

result = cv2.merge((bH,gH,rH))

chans = cv2.split(result)
colors = ("b","g","r")
plt.figure()
plt.xlabel("caitu Bins")
plt.ylabel("caitu pixels")
plt.xlim([0, 256])
for chan,color in zip(chans,colors):
    dst = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(dst,color=color)

plt.show()

# cv2.imshow("caitu Equalization:",result)
# cv2.waitKey()

