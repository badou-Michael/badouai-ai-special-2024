import cv2
import numpy as np
from matplotlib import pyplot as plt


# 读图 + 灰度化
img = cv2.imread("lenna.png", 1)    # 1: color image; 0: grayscale image； -1: unchanged
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 直方图均衡化
dst_gray = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst_gray], [0], None, [256], [0, 256])  #计算直方图 # 0:通道索引 # None:掩膜 # 256:直方图尺寸 # 0,256:像素值范围

plt.figure()
plt.hist(dst_gray.ravel(), 256)   # ravel() 将多维数组降为一维
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst_gray]))    # 横向拼接 gray 和 dst
cv2.waitKey(0)
cv2.destroyAllWindows()

# 彩色图像直方图均衡化

cv2.imshow("Source picture", img)

# 彩色图像均衡化，需要分别处理每个通道
(b, g, r) = cv2.split(img)
b_c = cv2.equalizeHist(b)
g_c = cv2.equalizeHist(g)
r_c = cv2.equalizeHist(r)

# 每一个通道的直方图
hist_b = cv2.calcHist([b_c], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g_c], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([r_c], [0], None, [256], [0, 256])

# 3个通道的直方图一起显示
plt.figure()
plt.plot(hist_b, color='b')
plt.plot(hist_g, color='g')
plt.plot(hist_r, color='r')
plt.xlim([0, 256])  # 设置x坐标轴范围
plt.title("Flattened Color Histogram")
plt.show()

# 合并每个通道
dst_bgr = cv2.merge((b_c, g_c, r_c))
cv2.imshow("Destination RGB Pic", dst_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()

