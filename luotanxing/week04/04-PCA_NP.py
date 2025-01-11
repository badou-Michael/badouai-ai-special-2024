# coding=utf-8

import numpy as np


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # 列数 特征向量个数
        self.features = X.shape[1]
        # 均值中心化 axis=0为每一列的均值
        X = X - X.mean(axis=0)
        # 求协方差
        conv = np.dot(X.T, X) / X.shape[0]
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(conv)
        # 获取排序后的索引
        idx = np.argsort(-eigenvalues)
        # 降维矩阵 切片，从维度A到维度B的前一个
        self.components_ = eigenvectors[:, idx[:self.n_components]]
        # 降维X
        Y = np.dot(X,self.components_ )
        return Y


# 调用
pca = PCA(n_components=2)
X = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
newX = pca.fit_transform(X)
print(newX)  # 输出降维后的数据
