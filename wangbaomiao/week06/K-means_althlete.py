# -*- coding: utf-8 -*-
# time: 2024/10/28 14:32
# file: K-means_althlete.py
# author: flame
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute

通过K-means算法对篮球比赛中的球员数据进行聚类，并将聚类结果可视化。

1. **数据准备**：
   - 定义一个包含多个二维坐标点的数据集 `X`，每个点代表一名球员在比赛中的表现，具体包括每分钟助攻数和每分钟得分数。
   - 打印数据集及其长度，以便确认数据是否正确加载。

2. **聚类**：
   - 使用 `KMeans` 类进行聚类，设定聚类中心数为3，表示将球员分为3个不同的类别。
   - 调用 `fit_predict` 方法对数据集进行聚类，并获取每个点所属的类别。
   - 打印聚类结果，显示每个点的类别标签。

3. **可视化**：
   - 提取数据集中所有点的X轴坐标和Y轴坐标。
   - 使用 `matplotlib` 库绘制散点图，颜色根据聚类结果区分，标记为'x'形状。
   - 设置图表标题、X轴标签、Y轴标签和图例，以便更好地理解图表内容。
   - 显示图表。
"""

# 初始化数据集X，包含多个二维坐标点，每个点代表一名球员在比赛中的表现
X = [[4.17, 2.21],
     [3.13, 1.69],
     [1.34, 5.42],
     [6.19, 4.69],
     [5.58, 4.84],
     [4.66, 5.64],
     [5.92, 4.37],
     [3.77, 1.84],
     [8.91, 7.16],
     [2.15, 3.00],
     [5.29, 3.83],
     [5.92, 1.92],
     [3.62, 7.16],
     [7.51, 3.11],
     [2.33, 5.92],
     [2.77, 1.44],
     [6.29, 1.44],
     [4.37, 2.16],
     [0.68, 4.69],
     [1.46, 5.48]]

# 打印数据集，确认数据是否正确加载
print("数据集：", X)

# 打印数据集长度，确认数据集的大小
print("数据集长度：", len(X))

"""
第二部分：聚类
"""
# 导入KMeans类，用于执行K-means聚类算法
from sklearn.cluster import KMeans

# 创建KMeans对象，设定聚类中心数为3，表示将球员分为3个不同的类别
kmeans = KMeans(n_clusters=3)

""" 对数据集X进行聚类，并获取每个点所属的类别标签
fit(X[, y, sample_weight])
作用：训练模型，找到最佳的聚类中心。
参数：
X：输入数据，通常是一个二维数组，形状为 (n_samples, n_features)。
y：可选参数，通常在无监督学习中不使用。
sample_weight：可选参数，用于指定每个样本的权重。
返回值：无返回值，但会更新 KMeans 对象的内部状态，包括聚类中心 cluster_centers_ 和每个样本的标签 labels_。
fit_predict(X[, y, sample_weight])
作用：训练模型并返回每个样本的聚类标签。
参数：
X：输入数据，通常是一个二维数组，形状为 (n_samples, n_features)。
y：可选参数，通常在无监督学习中不使用。
sample_weight：可选参数，用于指定每个样本的权重。
返回值：一个一维数组，形状为 (n_samples,)，表示每个样本的聚类标签。
fit_transform(X[, y, sample_weight])
作用：训练模型并返回转换后的数据。
参数：
X：输入数据，通常是一个二维数组，形状为 (n_samples, n_features)。
y：可选参数，通常在无监督学习中不使用。
sample_weight：可选参数，用于指定每个样本的权重。
返回值：一个二维数组，形状为 (n_samples, n_clusters)，表示每个样本到每个聚类中心的距离。
区别和作用
fit(X)：
主要用途：仅训练模型，找到最佳的聚类中心。
应用场景：当你只需要训练模型而不需要立即获取聚类标签或转换后的数据时使用。
fit_predict(X)：
主要用途：训练模型并立即返回每个样本的聚类标签。
应用场景：当你需要训练模型并立即获取每个样本的聚类标签时使用。这是最常用的组合方法，因为它一步完成训练和预测。
fit_transform(X)：
主要用途：训练模型并返回转换后的数据，通常用于降维或特征提取。
应用场景：当你需要训练模型并获取每个样本到每个聚类中心的距离时使用。这在某些情况下可以用于进一步的分析或可视化。
"""
result = kmeans.fit_predict(X)

# 打印聚类结果，显示每个点的类别标签
print("聚类中心：", result)

"""
第三部分：可视化绘图
"""
# 导入warnings库，用于处理警告信息
import warnings

# 忽略用户警告，以避免代码执行时产生不必要的警告信息
warnings.filterwarnings("ignore", category=UserWarning)

# 提取数据集中所有点的X轴坐标，即每分钟助攻数
x = [i[0] for i in X]

# 打印X轴坐标，确认提取是否正确
print(x)

# 提取数据集中所有点的Y轴坐标，即每分钟得分数
y = [i[1] for i in X]

# 打印Y轴坐标，确认提取是否正确
print(y)

# 导入matplotlib.pyplot库，用于绘制图表
import matplotlib.pyplot as plt

# 绘制散点图，颜色根据聚类结果区分，标记为'x'形状
plt.scatter(x, y, c=result, marker='x')

# 设置图表标题，描述图表内容
plt.title("K-means聚类-BasketBall Data")

# 设置X轴标签，描述X轴数据的含义
plt.xlabel("每分钟助攻数：assists_per_minute")

# 设置Y轴标签，描述Y轴数据的含义
plt.ylabel("每分钟得分数：points_per_minute")

# 设置图例，表示不同类别的标签
plt.legend(["A", "B", "C"])

# 显示图表
plt.show()
