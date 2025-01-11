# -*- coding: utf-8 -*-
# time: 2024/11/5 13:02
# file: 层次聚类.py
# author: flame

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from matplotlib import pyplot as plt
"""
导入所需的库
- `scipy.cluster.hierarchy` 提供了层次聚类的相关函数
- `matplotlib.pyplot` 用于绘图
对一组二维数据点进行层次聚类分析，并通过树状图（dendrogram）可视化聚类结果。
具体步骤如下：
1. 定义一组二维数据点 `X`。
2. 使用 `scipy.cluster.hierarchy.linkage` 函数进行层次聚类分析，生成聚类树 `Z`。
3. 使用 `scipy.cluster.hierarchy.fcluster` 函数根据聚类树 `Z` 和设定的距离阈值 `4` 进行聚类划分。
4. 使用 `matplotlib.pyplot` 绘制树状图，展示层次聚类的结果。
5. 打印聚类树数组 `Z` 并显示树状图。

定义一组二维数据点
- 每个数据点包含两个特征值，表示在二维空间中的位置
- 这些数据点将用于层次聚类分析
"""
X = [[1,2], [3,2], [4,4], [1,2], [1,3]]
"""
使用 `linkage` 函数进行层次聚类分析
- 参数 `X` 是输入的数据点
- 参数 `'ward'` 表示使用 Ward 方法，该方法旨在最小化每个聚类内的方差
- Ward 方法通过最小化聚类内部的方差来决定如何合并簇，从而形成层次结构
- 返回值 `Z` 是一个聚类树数组，记录了每次合并操作的信息
"""
Z = linkage(X, 'ward')

"""
使用 `fcluster` 函数根据聚类树 `Z` 和设定的距离阈值 `4` 进行聚类划分
- 参数 `Z` 是聚类树
- 参数 `4` 是最大距离阈值，表示当两个聚类之间的距离大于这个阈值时，它们将被划分为不同的聚类
- 参数 `'distance'` 表示按照距离阈值进行聚类划分
- 返回值 `f` 是一个数组，表示每个数据点所属的聚类标签
"""
f = fcluster(Z, 4, 'distance')

"""
创建一个新的图像，用于绘制树状图
- 参数 `figsize` 设置图像的大小，宽度为 5 英寸，高度为 3 英寸
- 这个图像将用于展示层次聚类的结果
"""
fig = plt.figure(figsize=(5,3))

"""
使用 `dendrogram` 函数绘制数据的树状图
- 参数 `Z` 是聚类树
- 树状图可以直观地展示层次聚类的过程和结果
"""
dn = dendrogram(Z)

"""
打印聚类树数组 `Z`，以查看聚类过程的详细信息
- 聚类树数组 `Z` 包含了每次合并操作的信息
- 每行的前两个元素是被合并的簇的索引，第三个元素是合并时的距离，第四个元素是合并后的簇中的样本数
"""
print(Z)

"""
使用 `plt.show()` 显示树状图
- 此函数会打开一个窗口，显示当前的图形
"""
plt.show()
