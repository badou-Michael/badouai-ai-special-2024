import cv2
import numpy as np
from matplotlib import  pyplot as plt
img = cv2.imread("lenna.png")
#转化为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
cv2.imshow("gray",gray)
cv2.imshow("dst",dst)
cv2.waitKey(0)

#直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
#创建一个新的图形窗口
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()
