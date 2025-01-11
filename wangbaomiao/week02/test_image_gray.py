# -*- coding: utf-8 -*-
# time: 2024/10/15 15:14
# file: test_image_gray.py
# author: flame
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread('lenna.png')
# 获取图片的高度和宽度
h,w = img.shape[:2]
# 初始化一个与原图大小相同的图
img_gray = np.zeros([h,w],img.dtype)
# 遍历图片的每个像素点，转换为灰度图
for i in range(h):
    for j in range(w):
        # 获取像素点的RGB值
        m = img[i,j]
        # 根据灰度转换公式计算灰度值
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

#print(m)
#print(img_gray)
print("imgage show gray: %s"%img_gray)
# 显示图片
# 显示灰度图像，并等待用户按键后关闭窗口
cv2.imshow("image show gray", img_gray)
cv2.waitKey(0)  # 等待时间，0表示无限等待，直到有按键输入
cv2.destroyAllWindows()  # 关闭所有窗口


# 显示图片
plt.subplot(221)
img = plt.imread('lenna.png')
# 使用 matplotlib 显示图像
plt.imshow(img)
plt.show()  # 显示图像并等待用户操作后关闭窗口

img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("----image gray----")
print(img_gray)

# 显示图片
plt.subplot(2, 2, 1)  # 指定2x2网格的第一格
plt.imshow(img)
plt.title('Original Image')  # 添加标题
#plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度

img_gray = rgb2gray(img)
plt.subplot(2, 2, 2)  # 指定2x2网格的第二格
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale Image')  # 添加标题
#plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.show()

print("----image gray----")
print(img_gray)


img_binary = np.where(img_gray >= 0.5,1,0)
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()