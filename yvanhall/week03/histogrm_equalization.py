# -*- coding: utf-8 -*-

import cv2
import numpy as np

# 读图
img = cv2.imread('lenna.png')

# 通道分解
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

#合并通道
result_img = cv2.merge((bH, gH, rH))
cv2.imshow('img', result_img)
cv2.waitKey(0)
