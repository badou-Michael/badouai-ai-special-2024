import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray



# 读取图片
img = cv2.imread("lenna.png")
h, w = img.shape[0:2]
# cv2.imshow("lenna_img", img)
# cv2.waitKey(0)
# print(img)
# print(img.shape)

# 灰度化
image_grey = np.zeros((h, w), img.dtype)
for i in range(h):
    for j in range(w):
        image_grey[i, j] = int(img[i, j][0]*0.11 + img[i, j][1]*0.59 + img[i, j][2]*0.3)


print(image_grey)
# print(image_grey.shape)

cv2.imshow("lenna_grey", image_grey)
cv2.waitKey(0)

# 读取图片
plt.subplot(221)
img = plt.imread('lenna.png')
plt.imshow(img)


# 灰度化
plt.subplot(222)
img_grey = rgb2gray(img)
plt.imshow(img_grey, cmap='gray')
print(img_grey)
print(img_grey.shape)

# opencv灰度化
# imggg = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
# img_opencv_grey = cv2.cvtColor(imggg, cv2.COLOR_BGR2GRAY)
# cv2.imshow("img_opencv_grey", img_opencv_grey)
# cv2.waitKey(0)
# res = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
# g = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
# cv2.imshow("灰度图", g)
# cv2.waitKey(0)


# 二值图
plt.subplot(223)
rows, cols = img_grey.shape
img_binary = np.zeros([rows, cols])
for i in range(rows):
    for j in range(cols):
        if img_grey[i, j] < 0.5:
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 1
print(img_binary)
print(img_binary.shape)
plt.imshow(img_binary, cmap='gray')


plt.show()
