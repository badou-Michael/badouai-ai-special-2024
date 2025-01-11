import cv2
import numpy as np
from matplotlib import pyplot as plt

# 灰度图像直方图
# 获取灰度图像
img = cv2.imread("lenna.png", 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 直方图均衡化
# dst = cv2.equalizeHist(gray)

# 均衡化之后的直方图
# hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
# plt.figure()
# plt.hist(dst.ravel(), 256)
# plt.show()

# 灰度图及均衡化之后的灰度图
# cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 彩色图像原图与直方图均衡化之后对比
cv2.imshow("origin image", img)

(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

cv2.imshow("histogram equalization", cv2.merge([bH, gH, rH]))
cv2.waitKey(0)

