# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("../lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度图的均衡化
dst = cv2.equalizeHist(gray)
cv2.imshow('equalize', dst)
cv2.waitKey()
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

# 彩色图的均衡化
b, g, r = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
dst2 = cv2.merge((bH, gH, rH))
cv2.imshow('equalize', dst2)
cv2.waitKey()