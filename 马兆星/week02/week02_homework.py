
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# 原图
img1 = plt.imread("lenna.png")
plt.subplot(221)
plt.imshow(img1)


# 灰度化
img = cv2.imread("lenna.png")
cv2.imshow('photo1',img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 灰度图
cv2.imshow('photo2',gray_img)
plt.subplot(222)
plt.imshow(gray_img, cmap='gray')

# 二值化，(127,255)为阈值
#retval, bit_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
bit_img = np.where(rgb2gray(img)>=0.5,1,0)

plt.subplot(223)
plt.imshow(bit_img, cmap='gray')

plt.show()

#
#
# cv2.waitKey(0)

