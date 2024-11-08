# -*- coding: gbk -*-
# time: 2024/10/26 14:13
# file: sobel_laplace_canny.py
# author: flame
import cv2
from matplotlib import pyplot as plt

# 读取图像文件 "lenna.png"，参数 1 表示以彩色图像读入，通道顺序为 BGR。
# 这里选择 1 是因为我们需要保留图像的彩色信息，以便后续将其转换为灰度图像。
img = cv2.imread("lenna.png", 1)

# 将读入的彩色图像转换为灰度图像。
# cv2.COLOR_RGB2GRAY 参数表示将 RGB 颜色空间转换为灰度颜色空间。
# 转换为灰度图像是为了简化图像数据，减少计算复杂度，并且边缘检测算法通常在灰度图像上效果更好。
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 使用 Sobel 算子在水平方向上检测图像边缘。
# cv2.Sobel 函数用于计算图像的一阶或二阶导数。
# img_gray: 输入图像，必须是单通道图像。
# cv2.CV_64F: 输出图像的数据类型，这里选择 64 位浮点型以避免溢出。
# 1: x 方向的导数阶数。
# 0: y 方向的导数阶数。
# ksize: Sobel 算子的卷积核大小，3 表示 3x3 的卷积核。
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)

# 使用 Sobel 算子在垂直方向上检测图像边缘。
# 参数与 img_sobel_x 相同，只是 x 和 y 方向的导数阶数互换了。
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

# 使用 Laplacian 算子进行图像边缘检测。
# cv2.Laplacian 函数用于计算图像的二阶导数。
# img_gray: 输入图像，必须是单通道图像。
# cv2.CV_64F: 输出图像的数据类型，这里选择 64 位浮点型以避免溢出。
# ksize: Laplacian 算子的卷积核大小，3 表示 3x3 的卷积核。
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)

# 使用 Canny 算子进行图像边缘检测。
# cv2.Canny 函数用于检测图像中的边缘。
# img_gray: 输入图像，必须是单通道图像。
# 100: 边缘检测的低阈值。
# 150: 边缘检测的高阈值。
# Canny 算子通过双阈值法来确定哪些边缘是真正的边缘。
img_canny = cv2.Canny(img_gray, 100, 150)

# 创建一个 2x3 的子图布局，并显示原始灰度图像。
# plt.subplot(231) 表示在 2x3 的子图布局中选择第 1 个位置。
# plt.imshow(img_gray, "gray") 用于显示图像，"gray" 表示使用灰度颜色映射。
# plt.title("original") 用于设置子图的标题。
plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("original")

# 创建一个 2x3 的子图布局，并显示 Sobel 算子在水平方向上的边缘检测结果。
# plt.subplot(232) 表示在 2x3 的子图布局中选择第 2 个位置。
# plt.imshow(img_sobel_x, "gray") 用于显示图像，"gray" 表示使用灰度颜色映射。
# plt.title("sobel_x") 用于设置子图的标题。
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("sobel_x")

# 创建一个 2x3 的子图布局，并显示 Sobel 算子在垂直方向上的边缘检测结果。
# plt.subplot(233) 表示在 2x3 的子图布局中选择第 3 个位置。
# plt.imshow(img_sobel_y, "gray") 用于显示图像，"gray" 表示使用灰度颜色映射。
# plt.title("sobel_y") 用于设置子图的标题。
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("sobel_y")

# 创建一个 2x3 的子图布局，并显示 Laplacian 算子的边缘检测结果。
# plt.subplot(234) 表示在 2x3 的子图布局中选择第 4 个位置。
# plt.imshow(img_laplace, "gray") 用于显示图像，"gray" 表示使用灰度颜色映射。
# plt.title("laplace") 用于设置子图的标题。
plt.subplot(234), plt.imshow(img_laplace, "gray"), plt.title("laplace")

# 创建一个 2x3 的子图布局，并显示 Canny 算子的边缘检测结果。
# plt.subplot(235) 表示在 2x3 的子图