
import cv2
import numpy as np
from matplotlib import pyplot as plt


# 获取灰度图像
img = cv2.imread("C:/Users/DMR/Desktop/1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 灰度直方图显示
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))

cv2.imshow("Gray Histogram Equalization", np.hstack([gray, dst]))
cv2.imshow("Color Histogram Equalization", np.hstack([img, result]))
cv2.waitKey(0)

