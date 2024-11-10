# -*- coding: utf-8 -*-
# time: 2024/10/18 16:23
# file: PCA_sklearn.py
# author: flame
# 导入numpy库，用于数值计算和数组操作
import numpy as np

# 导入sklearn库中的PCA模块，用于执行主成分分析（PCA）
from sklearn.decomposition import PCA

# 定义一个二维数组X，表示数据集，每行是一个样本，每列是一个特征
# 数据集中有6个样本，每个样本有4个特征
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])

# 创建一个PCA对象，指定降维后的维度为2
# n_components参数指定了降维后的目标维度
pca = PCA(n_components=2)

# 使用PCA对象对数据X进行降维处理
# fit_transform方法首先拟合数据，然后将数据转换到新的低维空间
newX = pca.fit_transform(X)

# 打印降维后的数据
# newX是一个二维数组，其中每个样本现在只有2个特征
print(newX)
