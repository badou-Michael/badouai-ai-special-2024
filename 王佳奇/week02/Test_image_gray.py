from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化一 浮点转换
plt.subplot(221)
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray_1 = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray_1[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

plt.imshow(img_gray_1, cmap='gray')


# 灰度化二 cv2直接转换
plt.subplot(222)
img_gray_2 = rgb2gray(img)
# img_gray_2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img_gray_2 = img
plt.imshow(img_gray_2, cmap='gray')


# 二值化一
plt.subplot(223)
rows, cols = img_gray_2.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray_2[i, j] <= 0.5):
            img_gray_2[i, j] = 0
        else:
            img_gray_2[i, j] = 1

plt.imshow(img_gray_2, cmap='gray')

# 二值化二
plt.subplot(224)
img_binary = np.where(img_gray_2 >= 0.5, 1, 0)
plt.imshow(img_binary, cmap='gray')
plt.show()
