# -*- coding: utf-8 -*-
# time: 2024/10/28 12:17
# file: k-means.py
# author: flame
import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt

"""
对一张名为 "lenna.png" 的图像进行 k-means 聚类，并将聚类结果与原图进行对比显示。
具体步骤如下：
1. 读取图像并转换为灰度图。
2. 获取图像的行数和列数，并将图像数据重塑为一维数组形式。
3. 将图像数据转换为 float32 类型，以满足 k-means 算法的要求。
4. 定义 k-means 算法的终止条件和初始中心点选择方式。
5. 使用 k-means 算法对图像数据进行聚类。
6. 将聚类结果的标签重塑为与原始图像相同的形状。
7. 设置 matplotlib 的字体为 SimHei，以支持中文显示。
8. 将原始图像和聚类后的图像并排显示。
"""

# 读取图像并转换为灰度图
# `cv2.imread` 函数用于读取图像文件，第二个参数 0 表示将图像转换为灰度图。
img = cv2.imread("lenna.png", 0)

# 打印图像形状，以检查其行和列
# `img.shape` 返回一个元组，表示图像的高度和宽度。
print(img.shape)

# 获取图像的行数和列数
# `img.shape` 返回的元组中，第一个元素是行数，第二个元素是列数。
rows, cols = img.shape[:]

# 重塑图像数据为一维数组形式，准备进行 k-means 聚类
# `reshape` 方法将图像数据重塑为 (rows * cols, 1) 形状的一维数组，以便于 k-means 算法处理。
data = img.reshape((rows * cols, 1))

# 将图像数据转换为 float32 类型，以满足 k-means 算法的要求
# k-means 算法要求输入数据为浮点类型，因此需要将数据转换为 float32 类型。
data = np.float32(data)

# 定义 k-means 算法的终止条件：当满足精度要求或达到最大迭代次数时停止
# `cv2.TERM_CRITERIA_EPS` 表示基于精度的终止条件，`cv2.TERM_CRITERIA_MAX_ITER` 表示基于最大迭代次数的终止条件。
# 第二个参数 10 表示最大迭代次数，第三个参数 1.0 表示精度阈值。
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 指定 k-means 算法的初始中心点选择方式
# `cv2.KMEANS_PP_CENTERS` 表示使用 k-means++ 算法选择初始中心点。
flags = cv2.KMEANS_PP_CENTERS

# 使用 k-means 算法对图像数据进行聚类
# `cv2.kmeans` 函数执行 k-means 聚类算法，参数分别为输入数据、聚类数、初始中心点、终止条件、尝试次数和初始中心点选择方式。
# 返回值包括 compactness（紧凑度）、labels（每个样本的聚类标签）和 centers（聚类中心）。
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 将聚类结果的标签重塑为与原始图像相同的形状
# `reshape` 方法将聚类标签重塑为与原始图像相同的形状 (rows, cols)。
dst = labels.reshape((img.shape[0], img.shape[1]))

# 设置 matplotlib 的字体为 SimHei，以支持中文显示
# `plt.rcParams` 用于设置 matplotlib 的全局参数，这里设置字体为 SimHei 以支持中文显示。
plt.rcParams['font.sans-serif'] = ['SimHei']

# 定义标题列表，用于显示原始图像和聚类后的图像
# `titles` 列表包含两个字符串，分别表示原始图像和聚类后图像的标题。
titles = ['原图', '聚类图像']

# 将原始图像和聚类后的图像存储在列表中
# `images` 列表包含两个图像，分别是原始图像和聚类后的图像。
images = [img, dst]

# 遍历图像列表，将原始图像和聚类后的图像并排显示
# `for` 循环遍历 `titles` 和 `images` 列表，使用 `plt.subplot` 创建子图，`plt.imshow` 显示图像，`plt.title` 设置标题。
# `plt.xticks([])` 和 `plt.yticks()` 用于隐藏 x 轴和 y 轴的刻度。
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'), plt.title(titles[i])
    plt.xticks([]), plt.yticks()

# 显示图像
# `plt.show` 函数用于显示所有创建的图像。
plt.show()
