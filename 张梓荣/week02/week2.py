"""
版本：
python                    3.7.12
opencv                    3.4.9
numpy                     1.21.5
matplotlib                3.5.3
"""
import cv2
import matplotlib.pyplot as plot
import numpy as np
from skimage.color import rgb2gray


print(np.floor(3.4))
print(np.ceil(3.4))
# 原图
img0 = cv2.imread('lenna.png')
plot.subplot(331)
img_temp = img0[:, :, ::-1]
plot.imshow(img_temp)

# 灰度调用接口实现
gray_img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
plot.subplot(334)
plot.imshow(gray_img0, cmap='gray')

# 灰度手工实现
h, w = img0.shape[:2]
gray_img1 = np.zeros([h, w], dtype=img0.dtype)
for i in range(h):
    for j in range(w):
        temp = img0[i, j]
        gray_img1[i, j] = int(temp[0] * 0.11 + temp[1] * 0.59 + temp[2] * 0.3)
plot.subplot(335)
plot.imshow(gray_img1, cmap='gray')

# 灰度调用接口归一化
plot.subplot(336)
plot.imshow(rgb2gray(img0), cmap='gray')



# 二值化
rows, cols = gray_img1.shape
img_binary = np.zeros([rows, cols], gray_img1.dtype)
for i in range(rows):
    for j in range(cols):
        if (gray_img1[i, j] <= 125):
            img_binary[i, j] = 1
        else:
            img_binary[i, j] = 0
plot.subplot(337)
plot.imshow(img_binary, cmap='gray')
plot.show()
