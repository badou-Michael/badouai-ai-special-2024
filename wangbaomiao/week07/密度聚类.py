# -*- coding: utf-8 -*-
# time: 2024/11/5 13:49
# file: 密度聚类.py
# author: flame
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN


"""
此段代码的主要目的是使用DBSCAN算法对鸢尾花数据集进行聚类，并将聚类结果可视化。
1. 首先加载鸢尾花数据集，并提取前4个特征。
2. 使用DBSCAN算法进行聚类，设置参数eps=0.4和min_samples=9。
3. 根据聚类结果，将不同标签的数据分别存储。
4. 使用matplotlib库绘制散点图，展示不同标签的数据点。
"""

""" 加载鸢尾花数据集，该数据集包含150个样本，每个样本有4个特征（萼片长度、萼片宽度、花瓣长度、花瓣宽度）。 """
iris = datasets.load_iris()

""" 提取数据集中的前4个特征，即萼片长度、萼片宽度、花瓣长度和花瓣宽度。 """
X = iris.data[:,:4]

""" 输出特征数据的形状，以验证数据是否正确加载。形状应为 (150, 4)。 """
print(X.shape)

""" 初始化DBSCAN模型，eps参数表示两个点被认为是邻近点的最大距离，min_samples参数表示一个点被定义为核心点所需的最小邻近点数。 """
dbscan = DBSCAN(eps=0.4, min_samples=9)

""" 使用DBSCAN模型对特征数据进行聚类。 """
dbscan.fit(X)

""" 获取聚类后的标签，每个样本会被分配一个标签，-1表示噪声点。 """
label_pred = dbscan.labels_

""" 根据聚类结果，将不同标签的数据分别存储。 """
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

""" 使用matplotlib库绘制散点图，展示不同标签的数据点。 """
plt.scatter(x0[:,0], x0[:,1], c='red', marker='o', label='label0')
""" 绘制标签为0的数据点，颜色为红色，标记为圆圈。 """

plt.scatter(x1[:,0], x1[:,1], c='green', marker='*', label='label1')
""" 绘制标签为1的数据点，颜色为绿色，标记为星号。 """

plt.scatter(x2[:,0], x2[:,1], c='blue', marker='+', label='label2')
""" 绘制标签为2的数据点，颜色为蓝色，标记为加号。 """

""" 设置x轴标签为“萼片长度”。 """
plt.xlabel('sepal length')

""" 设置y轴标签为“萼片宽度”。 """
plt.ylabel('sepal width')

""" 添加图例，位置在左上角。 """
plt.legend(loc=2)

""" 显示图形。 """
plt.show()


