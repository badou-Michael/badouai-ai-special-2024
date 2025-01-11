# -*- coding: utf-8 -*-
# 实现直方图均衡化

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(img_gray)

cv2.imshow("hist_dst",np.hstack([img_gray,dst]))
cv2.waitKey(0)
