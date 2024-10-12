
##### 这是分别实现的草稿版本，需要单独开启代码执行。

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

### 使原图效果：用matplotlib展示
# ori_img = plt.imread("lena.png")
# plt.imshow(ori_img)
# plt.show()
# print(ori_img)

### 作业一：实现灰度化
## 方法1：使用opencv和numpy根据灰度化公式进行换算
# img = cv2.imread("lena.png")
# # print(img.shape)  # 返回一个元组，包含了数组（图像）的所有维度。第一个维度是图像的高度（行数），第二个维度是图像的宽度（列数）。如果图像是彩色的，那么第三个维度是颜色通道（通常是3个通道，对应于RGB）。
# h, w = img.shape[:2]  # 获取数组（图像）的高度（行数）和宽度（列数）。
# img_gray = np.zeros(shape=[h, w], dtype=img.dtype)  # 创建一个与原始图像具有相同尺寸和数据类型的全零数组
# for i in range(h):
#     for j in range(w):
#         m = img[i, j]  # 获取位于第 i 行、第 j 列的像素值。
#         # print(m)  # 在RGB图像中，m 是一个包含三个元素的数组（或列表），分别对应于该像素的红色、绿色和蓝色通道的强度值。
#         img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
#
# # print(img_gray)
# cv2.imshow(winname="opencv&numpy img_gray", mat=img_gray)  # 使用opencv展示新图片
# cv2.waitKey(0)  # 等待用户按键
# cv2.destroyAllWindows()  # 销毁所有 OpenCV 窗口

## 方法2：使用opencv的属性
# img = cv2.imread("lena.png")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow(winname="opencv img_gray", mat=img_gray)  # 使用opencv展示新图片
# cv2.waitKey(0)  # 等待用户按键
# cv2.destroyAllWindows()  # 销毁所有 OpenCV 窗口

# 方法3：使用skimage
# img = plt.imread("lena.png")
# img_gray = rgb2gray(img)
# plt.imshow(img_gray, cmap='gray')  # cmap 参数代表颜色映射（colormap），它是一个预定义的颜色空间，用于将数据值映射到颜色。cmap 不仅可以用于灰度图像，还可以用于伪彩色图像，这在显示多维数据或强调数据中的特定特征时非常有用。
# plt.show()

### 作业二：实现二值化
# img = plt.imread("lena.png")
# img_gray = rgb2gray(img)
# img_binary = np.where(img_gray >= 0.5, 1, 0)  # 将灰度图像转换为二值图像。where 函数是一个条件函数，它根据给定条件返回输入数组中的元素，或者在条件不满足时返回另一个值。
# print(img_binary)
# plt.imshow(img_binary, cmap='gray')
# plt.show()
