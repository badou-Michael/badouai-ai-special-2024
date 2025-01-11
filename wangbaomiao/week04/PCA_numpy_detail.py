# -*- coding: utf-8 -*-
# time: 2024/10/18 14:01
# file: PCA_numpy_detail.py
# author: flame
"""
使用PCA求样本矩阵K阶降维矩阵Z
"""
import numpy as np

"""

"""

class CPCA(object):
    """
    CPCA 类用于执行主成分分析 (PCA) 并将样本矩阵降维到指定的维度。

    主要步骤包括：
    1. 初始化类实例，设置样本矩阵和降维后的特征数量。
    2. 计算样本矩阵的中心化矩阵。
    3. 计算中心化矩阵的协方差矩阵。
    4. 计算协方差矩阵的特征值和特征向量，并选择前K个特征向量作为降维转换矩阵。
    5. 使用降维转换矩阵将样本矩阵降维。
    """

    def __init__(self, X, K):
        """
        用PCA求样本矩阵K阶的降维矩阵Z

        参数:
        X (numpy.ndarray): 样本矩阵，形状为 (m, n)，其中 m 是样本数量，n 是特征数量。
        K (int): 降维后的特征数量。

        属性:
        X (numpy.ndarray): 输入的样本矩阵。
        K (int): 降维后的特征数量。
        centerX (numpy.ndarray): 中心化的样本矩阵。
        C (numpy.ndarray): 样本矩阵的协方差矩阵。
        U (numpy.ndarray): 降维转换矩阵。
        Z (numpy.ndarray): 降维后的样本矩阵。
        """
        self.X = X  # 设置样本矩阵
        self.K = K  # 设置降维后的特征数量
        self.centerX = []  # 初始化中心化矩阵
        self.C = []  # 初始化协方差矩阵
        self.U = []  # 初始化降维转换矩阵
        self.Z = []  # 初始化降维后的样本矩阵

        # 在初始化时依次计算中心化矩阵、协方差矩阵、降维转换矩阵和降维矩阵
        self.centerX = self._centralize()  # 计算中心化矩阵
        self.C = self._cov()  # 计算协方差矩阵
        self.U = self._U()  # 计算降维转换矩阵
        self.Z = self._Z()  # 计算降维后的样本矩阵

    def _centralize(self):
        """
        矩阵X的中心化

        返回:
        centerX (numpy.ndarray): 中心化的样本矩阵。
        """
        print("样本矩阵X:\n", self.X)  # 打印原始样本矩阵
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 计算每个特征的均值
        centerX = self.X - mean  # 从每个样本中减去均值，实现中心化
        print("样本矩阵X的中心化centerX:\n", centerX)  # 打印中心化后的样本矩阵
        return centerX  # 返回中心化后的样本矩阵

    def _cov(self):
        """
        求样本矩阵X的协方差矩阵C

        返回:
        C (numpy.ndarray): 协方差矩阵。
        """
        ns = np.shape(self.centerX)[0]  # 获取样本数量
        C = np.dot(self.centerX.T, self.centerX) / (ns - 1)  # 计算协方差矩阵
        print("样本矩阵X的协方差矩阵C:\n", C)  # 打印协方差矩阵
        return C  # 返回协方差矩阵

    def _U(self):
        """
        求X的降维转换矩阵U，形状为 (n, k)，其中 n 是特征维度总数，k 是降维后的特征维度

        返回:
        U (numpy.ndarray): 降维转换矩阵。
        """
        a, b = np.linalg.eig(self.C)  # 计算协方差矩阵的特征值和特征向量
        print("X的协方差矩阵C的特征值a:\n", a)  # 打印特征值
        print("X的协方差矩阵C的特征向量b:\n", b)  # 打印特征向量

        # 将特征值按降序排列，并选择前K个特征向量
        ind = np.argsort(-1 - a)  # 按特征值降序排列索引
        UT = [b[:, ind[i]] for i in range(self.K)]  # 选择前K个特征向量
        U = np.transpose(UT)  # 转置特征向量矩阵
        print("样本矩阵X的降维转换矩阵U:\n", U)  # 打印降维转换矩阵
        return U  # 返回降维转换矩阵

    def _Z(self):
        """
        求样本矩阵X的降维矩阵Z

        返回:
        Z (numpy.ndarray): 降维后的样本矩阵。
        """
        Z = np.dot(self.X, self.U)  # 通过降维转换矩阵将样本矩阵降维
        print("X shape:\n", np.shape(self.X))  # 打印原始样本矩阵的形状
        print("U shape:\n", np.shape(self.U))  # 打印降维转换矩阵的形状
        print("Z shape:\n", np.shape(Z))  # 打印降维后样本矩阵的形状
        print("Z:\n", Z)  # 打印降维后的样本矩阵
        return Z  # 返回降维后的样本矩阵

if __name__ == '__main__':
    """
    使用10个样本，每个样本有3个特征的数据集进行PCA降维
    """
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])  # 定义样本矩阵
    K = np.shape(X)[1] - 1  # 降维后的特征数量为原特征数量减1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)  # 打印样本集

    # 使用CPCA类进行PCA降维
    pca = CPCA(X, K)  # 创建CPCA对象并进行降维
