import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray_img)
#绘制灰度图像的直方图
hist = cv2.calcHist([gray_img], [0], None, [256], [0,256])
plt.plot(hist)
plt.show()

#直方图均衡化
hist_img = cv2.equalizeHist(gray_img)
cv2.imshow("hist", hist_img)

#彩色图像直方图均衡化
(b, g, r) = cv2.split(img)
hist_b = cv2.equalizeHist(b)
hist_g = cv2.equalizeHist(g)
hist_r = cv2.equalizeHist(r)
hist_color = cv2.merge((hist_b, hist_g, hist_r))
cv2.imshow("color", hist_color)



cv2.waitKey(0)