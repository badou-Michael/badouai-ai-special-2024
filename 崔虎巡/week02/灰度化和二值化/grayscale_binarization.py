"""
@author: cuihuxun
name：崔虎巡
彩色图像的灰度化、二值化
"""
from click import pause
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

#***********灰度化***********#
img = cv2.imread('lenna.png',cv2.IMREAD_COLOR)
high,wide = img.shape[:2]
img_gray = np.zeros([high,wide],img.dtype)

for i in range(high):
    for j in range(wide):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print(m)
print(img_gray)
cv2.imshow("imag_gray",img_gray)
cv2.waitKey(10000) #等待键盘输入

#***********二值化***********#
rows,cols = img_gray.shape
img_bin = np.zeros([rows,cols],img_gray.dtype)
for i in range(rows):
    for j in range(cols):
        if img_gray[i,j] >=128:
            img_bin[i,j] = 255
        else:
            img_bin[i,j] = 0

cv2.imshow("img_bin",img_bin)
cv2.waitKey(10000) #等待键盘输入
cv2.destroyAllWindows()
