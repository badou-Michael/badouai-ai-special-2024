# coding=utf-8

import numpy as np


class PCA(object):
    def __init__(self, n_components):
        self.components_ = None
        self.covariance = None
        self.n_features_ = None
        self.n_components = n_components

    def fit_transform(self, x):
        """
        中心化 -> 计算协方差矩阵 -> 求协方差矩阵的特征值和特征向量 -> 对特征值进行排序 -> 获得降维矩阵 -> 对原始函数进行降维
        :param x: 原始矩阵
        :return: 返回降维后的矩阵
        """
        """
        1.获取原始矩阵特征数量
        x 表示一个矩阵
        x.shape 返回一个包含两个整数的元组，第一个整数表示样本数量，第二个整数表示特征数量。因此，x.shape[1] 获取的就是特征数量
        """
        self.n_features_ = x.shape[1]
        """
        2.矩阵中心化 -> 原始矩阵减去均值
        x 是一个NumPy数组。
        mean 是NumPy数组对象的一个方法，用于计算数组的均值。
        axis=0 参数指定了计算均值的轴。在二维数组中，axis=0 表示沿着列方向计算均值，即对每一列的元素求和并除以该列的元素个数。
        """
        x = x - x.mean(axis=0)
        """
        3.计算协方差矩阵
        x.T：这是 x 的转置矩阵。转置操作将矩阵的行变为列，列变为行。
        np.dot(x.T, x)：这是矩阵乘法操作。它计算了 x 的转置矩阵与 x 本身的点积。结果是一个对称矩阵，其中每个元素 (i, j) 表示 x 中第 i 列和第 j 列的协方差。
        / x.shape[0]：这是对上述结果进行归一化处理。x.shape[0] 返回 x 的行数，即样本数量。通过除以样本数量，我们得到了协方差矩阵的样本均值，而不是简单的样本协方差。
        """
        self.covariance = np.dot(x.T, x) / x.shape[0]
        """
        4.求协方差矩阵的特征值和特征向量
        如果self.covariance是一个3x3的协方差矩阵
        eig_vals将是一个包含特征值的数组
        eig_vectors将是一个包含特征向量的矩阵，其中每一列都是一个特征向量。
        linalg.eig函数只能用于方阵
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance)
        """
        5.对特征值进行排序
        argsort函数对取负后的eig_vals进行排序，返回的是排序后的索引数组
        """
        idx = np.argsort(-eigenvalues)
        """
        6.获取降维矩阵
        , 是用于分隔矩阵的行和列的
        : 表示选取所有行，idx[:self.n_components] 表示选取指定的列
        """
        self.components_ = eigenvectors[:, idx[:self.n_components]]
        """
        7.对原始函数进行降维
        """
        return np.dot(x, self.components_)


# 调用
pca = PCA(n_components=2)
test_matrix = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
newX = pca.fit_transform(test_matrix)
print(newX)  # 输出降维后的数据
