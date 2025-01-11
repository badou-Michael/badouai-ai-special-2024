# -*- coding: utf-8 -*-
# time: 2024/10/28 14:56
# file: K-means_RGB.py
# author: flame
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
通过K-means聚类算法对图像进行颜色量化，并展示不同K值下的聚类结果。
具体步骤如下：
1. 读取图像并打印其形状。
2. 将图像的二维像素转换为一维数组，以便进行K-means聚类。
3. 定义K-means算法的终止条件和更新簇中心的方法。
4. 使用不同的K值进行K-means聚类。
5. 将聚类中心转换为uint8类型，并将一维数组重新转换为二维图像。
6. 将图像转换为灰度图，以便显示。
7. 设置中文字体以支持中文标题。
8. 将图像显示在2x3的网格中。
"""

# 读取图像文件 "lenna.png" 并将其存储在变量 img 中
img = cv2.imread("lenna.png")

# 打印图像的形状，即高度、宽度和通道数
# 这有助于了解图像的基本信息
print(img.shape)

# 将图像的二维像素转换为一维数组，以便进行K-means聚类
# -1 表示自动计算行数，3 表示每行有3个元素（BGR三个通道）
data = img.reshape((-1, 3))

# 将数据类型转换为浮点型，以满足K-means算法的要求
# K-means算法需要浮点型数据来进行计算
data = np.float32(data)

# 打印转换后数据的形状，验证转换是否正确
print(data.shape)

# 定义K-means算法的终止条件和更新簇中心的方法
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER 表示同时使用最大迭代次数和精度作为终止条件
# 10 表示最大迭代次数，1.0 表示所需的精度
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 指定K-means算法初始化簇中心的方法
# cv2.KMEANS_RANDOM_CENTERS 表示随机选择初始簇中心
flags = cv2.KMEANS_RANDOM_CENTERS

# 使用不同的K值进行K-means聚类
# compactness 是紧凑度，表示每个点到其最近簇中心的距离平方和
# label_2 是每个点所属的簇标签
# center_2 是每个簇的中心
compactness, label_2, center_2 = cv2.kmeans(data, 2, None, criteria, 10, flags)

# 使用K=4进行K-means聚类
compactness, label_4, center_4 = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 使用K=8进行K-means聚类
compactness, label_8, center_8 = cv2.kmeans(data, 8, None, criteria, 10, flags)

# 使用K=16进行K-means聚类
compactness, label_16, center_16 = cv2.kmeans(data, 16, None, criteria, 10, flags)

# 使用K=64进行K-means聚类
compactness, label_64, center_64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

# 将聚类中心转换为uint8类型，以便重新构建图像
# uint8 是无符号8位整数，适用于图像数据
center_2 = np.uint8(center_2)

# 将一维数组重新转换为二维图像
# label_2.flatten() 将标签数组展平为一维数组
# center_2[label_2.flatten()] 根据标签选择对应的簇中心
# res.reshape((img.shape)) 将一维数组重新转换为与原图像相同的形状
res = center_2[label_2.flatten()]
dst_2 = res.reshape((img.shape))

# 对K=4的情况重复上述步骤
center_4 = np.uint8(center_4)
res = center_4[label_4.flatten()]
dst_4 = res.reshape((img.shape))

# 对K=8的情况重复上述步骤
center_8 = np.uint8(center_8)
res = center_8[label_8.flatten()]
dst_8 = res.reshape((img.shape))

# 对K=16的情况重复上述步骤
center_16 = np.uint8(center_16)
res = center_16[label_16.flatten()]
dst_16 = res.reshape((img.shape))

# 对K=64的情况重复上述步骤
center_64 = np.uint8(center_64)
res = center_64[label_64.flatten()]
dst_64 = res.reshape((img.shape))

# 将图像转换为灰度图，以便显示
# cv2.COLOR_BGR2GRAY 将BGR图像转换为灰度图
dst_2 = cv2.cvtColor(dst_2, cv2.COLOR_BGR2GRAY)
dst_4 = cv2.cvtColor(dst_4, cv2.COLOR_BGR2GRAY)
dst_8 = cv2.cvtColor(dst_8, cv2.COLOR_BGR2GRAY)
dst_16 = cv2.cvtColor(dst_16, cv2.COLOR_BGR2GRAY)
dst_64 = cv2.cvtColor(dst_64, cv2.COLOR_BGR2GRAY)

# 设置中文字体以支持中文标题
# plt.rcParams['font.sans-serif'] = ['SimHei'] 设置Matplotlib的默认字体为SimHei，支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']

# 定义图像显示的标题和图像列表
# title 存储每个图像的标题
# imgs 存储每个图像的数据
title = [u'原始图像', u'K-means K=2', u'K-means K=4', u'K-means K=8', u'K-means K=16', u'K-means K=64']
imgs = [img, dst_2, dst_4, dst_8, dst_16, dst_64]

# 遍历标题和图像列表，将图像显示在2x3的网格中
# plt.subplot(2, 3, i+1) 创建一个2x3的子图网格，并选择第i+1个子图
# plt.imshow(imgs[i], 'gray') 显示第i个图像，使用灰度模式
# plt.title(title[i]) 设置子图的标题
# plt.xticks([]), plt.yticks([]) 去掉坐标轴刻度
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(imgs[i], 'gray')
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])

# 显示所有图像
plt.show()
