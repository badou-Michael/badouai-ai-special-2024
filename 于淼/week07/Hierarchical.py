# 层次聚类

import cv2
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import matplotlib.pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
# 执行层次聚类
Z = linkage(X,'ward')       # 这里使用Ward方法（'ward'）对数据集X执行层次聚类。
# 执行扁平化聚类
f = fcluster(Z,4,'distance')    # 4是指将数据点分为了4组，使用'distance'作为确定何时合并或分割聚类的准则
# 绘制一个5*3的图形窗口
fig = plt.figure(figsize=(5,3))
# 绘制了Z的树状图（dendrogram）
dng = dendrogram(Z)

print(Z)
plt.show()
