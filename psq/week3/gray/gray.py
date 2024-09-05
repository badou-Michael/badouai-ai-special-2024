import cv2
import os
import numpy as np

img=cv2.imread("D:\\galproject\\BDAN\\game\\images\\3.png")
print(img.shape)
array_2d=np.zeros((1024,1024),img.dtype)
img_gray=array_2d
for i in range(0,1024):
	for j in range(1024):
		bgr=img[i,j]
		img_gray[i,j]=(0.11*bgr[0]+0.59*bgr[1]+0.3*bgr[2])
#print(img_gray)
cv2.imshow("gray",img_gray)
cv2.waitKey(0)
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Otsu Threshold', thresh)
cv2.waitKey(0)
