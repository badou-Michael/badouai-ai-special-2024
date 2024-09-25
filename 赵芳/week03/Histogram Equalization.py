import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread("lenna.png", 1) # 1表示以BGR读取
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换灰度图像
#cv2.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256]) # 计算均衡化后的图像的直方图
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

# 显示原图与均衡化后的图像
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)


# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img) # 原图

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b) # 蓝色通道
gH = cv2.equalizeHist(g) # 绿色通道
rH = cv2.equalizeHist(r) # 红色通道
# 合并通道
result = cv2.merge((bH, gH, rH)) # 将均衡化后的通道合并成彩色图像
cv2.imshow("dst_rgb", result) # 显示均衡化后的彩色图像
cv2.waitKey(0)
