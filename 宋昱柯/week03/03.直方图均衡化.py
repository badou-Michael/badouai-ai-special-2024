import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("practice/cv/week03/lenna.png", 0)
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray',gray)

# 直方图均衡化
dst = cv2.equalizeHist(img)
# 计算直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
# print(hist)
# plt.plot(hist)

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()


cv2.imshow("Histogram Equalization", np.hstack([img, dst]))
cv2.waitKey(0)
