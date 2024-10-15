# -*- coding: utf-8 -*-
"""
@author: Michael

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化
img = cv2.imread("../sea.jpg")
# 遍历加权平均给图片做灰度处理， 太慢了
# h, w = img.shape[:2]  # 获取图片的high和wide
# img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
# for i in range(h):
#     for j in range(w):
#         m = img[i, j]  # 取出当前high和wide中的BGR坐标
#         img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像

# 使用opencv 转换颜色空间
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
# 这里表达式是向量化操作，底层一次性计算，不遍历整个数组所以很快
img_gray = 0.11*b + 0.59*g + 0.3*r
img_gray = img_gray.astype(np.uint8)  # 转换为无符号 8 位整数
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)

plt.subplot(221)
img = plt.imread("../sea.jpg")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

# 二值化
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1

img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')


#通过计算图像的全局均值或高斯加权均值来选择阈值。
# cv2.threshold函数的返回值是一个包含两个元素的元组，第一个元素是阈值，第二个元素是二值化后的图像,使用下划线 _ 来忽略阈值，只保留二值化结果 img_binary2。
mean_val = np.mean(img_gray)
_, img_binary2 = cv2.threshold(img_gray, mean_val, 255, cv2.THRESH_BINARY)
print("-----imge_binary2------")
print(img_binary2)
print(img_binary2.shape)

plt.subplot(224)
plt.imshow(img_binary2, cmap='gray')
plt.show()
