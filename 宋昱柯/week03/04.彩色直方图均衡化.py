import cv2
import numpy
from matplotlib import pyplot as plt

img = cv2.imread("practice/cv/week03/lenna.png",1)
(b,g,r)=cv2.split(img)
b_hist=cv2.equalizeHist(b)
g_hist=cv2.equalizeHist(g)
r_hist=cv2.equalizeHist(r)

#合并通道
res=cv2.merge((b_hist,g_hist,r_hist))

cv2.imshow("det_img",res)
cv2.waitKey(0)