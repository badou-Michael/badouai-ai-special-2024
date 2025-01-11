# -*- coding: utf-8 -*-
# time: 2024/10/28 11:08
# file: WrapPerspection.py
# author: flame
import cv2
import numpy as np

"""
1. 定义源点和目标点，用于计算透视变换矩阵。
2. 计算透视变换矩阵，使用 OpenCV 的 getPerspectiveTransform 函数。
3. 读取图像并创建副本，以便在副本上进行变换操作。
4. 应用透视变换矩阵对图像进行变换，使用 OpenCV 的 warpPerspective 函数。
5. 显示原始图像和变换后的图像，使用 OpenCV 的 imshow 函数。
"""

# 定义源点和目标点
# 源点 src 是原始图像中的四个角点，目标点 dst 是变换后图像中的四个角点。
# 使用 np.float32 确保坐标是浮点数，以提高透视变换的精度。
src = np.float32([[280, 650], [1090, 650], [1720, 1200], [55, 1200]])
dst = np.float32([[280, 650], [1090, 650], [1720, 1200], [55, 1200]])

# 计算透视变换矩阵
# 使用 OpenCV 的 cv2.getPerspectiveTransform 函数计算透视变换矩阵 m。
# 这个函数接受源点和目标点作为输入，返回一个 3x3 的透视变换矩阵。
m = cv2.getPerspectiveTransform(src, dst)

# 打印透视变换矩阵
# 打印矩阵 m 以便调试和验证计算结果。
print("WrapMatrix: %s" % m)

# 读取图像并创建副本
# 使用 OpenCV 的 cv2.imread 函数读取名为 "photo1.jpg" 的图像文件，并将其存储在变量 img 中。
img = cv2.imread("photo1.jpg")

# 创建 img 的副本 copy_img，使用 copy 方法是为了避免在后续操作中修改原始图像。
copy_img = img.copy()

# 打印原始图像的形状
# shape 属性返回一个包含图像高度、宽度和通道数的元组。
# 打印图像形状是为了验证图像是否成功读取，并了解其尺寸和通道信息。
print(img.shape)

# 打印副本图像的形状
# 由于 copy_img 是 img 的副本，因此它们的形状应该相同。
# 验证副本图像的形状是否与原始图像一致，确保复制操作正确无误。
print(copy_img.shape)

# 应用透视变换
# 使用 OpenCV 的 cv2.warpPerspective 函数对 copy_img 进行透视变换。
# m 是之前计算的透视变换矩阵，(280, 650) 是输出图像的尺寸。
# 透视变换将图像从一个平面投影到另一个平面，根据透视变换矩阵 m 对图像进行变形。
wrapMatrix = cv2.warpPerspective(copy_img, m, (280, 650))

# 显示原始图像
# 使用 OpenCV 的 cv2.imshow 函数显示原始图像 img，窗口标题为 "src"。
# 显示原始图像以便与变换后的图像进行对比。
cv2.imshow("src", img)

# 显示变换后的图像
# 使用 OpenCV 的 cv2.imshow 函数显示变换后的图像 wrapMatrix，窗口标题为 "wrapMatrix"。
# 显示变换后的图像以验证透视变换的效果。
cv2.imshow("wrapMatrix", wrapMatrix)

# 等待用户按键
# 使用 OpenCV 的 cv2.waitKey 函数等待用户按键。
# 如果没有按键事件，程序会一直等待。
# cv2.waitKey 用于保持窗口打开，直到用户按下任意键，否则窗口会立即关闭。
cv2.waitKey()
