# coding: utf-8
"""
    K-Means
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
# 导入 OpenCV、NumPy 和 Matplotlib 库，用于图像处理和可视化

# 读取灰度图像
img = cv2.imread("lenna.png", 0)
print("灰度图形状：",img.shape)
# 使用 OpenCV 的 imread 函数读取名为 "lenna.png" 的图像，并以灰度模式读取（参数 0）。
# 然后打印图像的形状，通常是 (height, width)。

image = cv2.imread("lenna.png", 1)
print("彩图形状：",img.shape)


rows, cols = img.shape[:]
# 获取图像的行数和列数。

data = img.reshape((rows * cols, 1))
data = np.float32(data)
# 将图像数据重塑为一维数组，每个元素是一个像素值。
# 然后将数据转换为 32 位浮点数类型。

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 设置 K-Means 算法的停止条件，这里是当迭代次数达到 10 或者达到指定的精度（EPS）时停止。

flags = cv2.KMEANS_RANDOM_CENTERS
# 设置 K-Means 算法的初始化方式为随机选择初始聚类中心。

ret, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
# 执行 K-Means 算法，data 是输入数据，3 是要分成的聚类数，None 表示没有初始中心，
# criteria 是停止条件，10 是迭代次数，flags 是初始化方式。
# 返回值 ret 是紧密度指标，labels 是每个像素点所属的聚类标签，centers 是聚类中心。

dst = labels.reshape((img.shape[0], img.shape[1]))
# 将标签数组重塑为与原始图像相同的形状。

# 设置中文字体为黑体
plt.rcParams["font.sans-serif"] = ["SimHei"]

# 显示原始图像和聚类后的图像
plt.subplot(131), plt.imshow(img, 'gray'), plt.title('原始图像')
# 创建一个 1 行 3 列的子图布局，并选择第一个子图。
# 在这个子图中显示原始灰度图像，并设置标题为 '原始图像'。

plt.subplot(132), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('原始彩色图像')
# 选择第二个子图，将原始图像从 BGR 格式转换为 RGB 格式后显示，并设置标题为 '原始彩色图像'。

plt.subplot(133), plt.imshow(dst, 'gray'), plt.title('聚类后图像')
# 选择第三个子图，显示聚类后的图像，并设置标题为 '聚类后图像'。

plt.show()
# 显示所有的子图。
