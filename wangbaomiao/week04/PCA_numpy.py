# -*- coding: utf-8 -*-
# time: 2024/10/18 16:10
# file: PCA_numpy.py
# author: flame
import numpy as np

class PCA():
    """
    主成分分析（PCA）类用于降维。

    参数:
    n_components (int): 所需的主成分（即降维后的维度）数量。
    """

    def __init__(self, n_components=None):
        """
        初始化PCA类，设置所需的主成分数量。

        参数:
        n_components (int): 所需的主成分数量。如果为None，则保留所有主成分。
        """
        self.n_components = n_components  # 存储所需的主成分数量

    def fit_transform(self, X):
        """
        计算数据集的主成分并进行降维处理。

        参数:
        X (numpy.ndarray): 输入的数据矩阵，每一行代表一个样本，每一列代表一个特征。

        返回:
        numpy.ndarray: 降维后的数据矩阵。
        """
        # 计算特征数量
        self.n_feature_ = X.shape[1]  # 获取输入数据的特征数量，shape[1]表示列数，即特征数量

        # 对数据进行中心化处理
        X = X - X.mean(axis=0)  # 每个特征减去其均值，使数据均值为0，axis=0表示按列计算均值

        # 计算数据的协方差矩阵
        self.covariance = np.dot(X.T, X) / X.shape[0]  # 计算协方差矩阵，公式为 (X^T * X) / n_samples，其中X^T是X的转置

        # 计算协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)  # 使用numpy的linalg模块计算协方差矩阵的特征值和特征向量
        # eig_vals 是特征值数组，eig_vectors 是对应的特征向量矩阵

        # 对特征值进行降序排序，并选择前n_components个特征向量
        idx = np.argsort(-eig_vals)  # 对特征值按降序排序，返回排序后的索引，-eig_vals 表示取负值以便按降序排序
        self.n_components_ = eig_vectors[:, idx[:self.n_components]]  # 选择前n_components个特征向量，idx[:self.n_components] 取前n_components个索引

        # 使用选定的特征向量对原始数据进行降维
        return np.dot(X, self.n_components_)  # 将中心化后的数据乘以选定的特征向量，得到降维后的数据

# 调用PCA类进行降维处理
pca = PCA(n_components=2)  # 创建PCA对象，指定降维后的维度为2
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
newX = pca.fit_transform(X)  # 对数据进行降维处理
print(newX)  # 打印降维后的数据
