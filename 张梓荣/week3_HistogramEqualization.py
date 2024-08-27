"""
直方图均衡化
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
plt.figure()
plt.hist(dst.ravel(), 256)
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))

# 彩色图均衡化
(b, g, r) = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)
result = cv2.merge((bh, gh, rh))
cv2.imshow("result", result)
cv2.imshow("img", img)
plt.show()
cv2.waitKey(0)
