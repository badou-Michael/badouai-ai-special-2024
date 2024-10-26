# -*- coding: utf-8 -*-
# time: 2024/10/18 14:01
# file: PCA_numpy_detail.py
# author: flame
"""
使用PCA求样本矩阵K阶降维矩阵Z
"""
import numpy as np

class CPCA(object):
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
        self.X = X  # 样本矩阵
        self.K = K  # K阶降维矩阵的K值
        self.centerX = []  # 矩阵X的中心化
        self.C = []  # 样本矩阵的协方差矩阵C
        self.U = []  # 样本矩阵的降维转换矩阵U
        self.Z = []  # 样本矩阵的降维矩阵Z

        # 在初始化时依次计算中心化矩阵、协方差矩阵、降维转换矩阵和降维矩阵
        self.centerX = self._centralize()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralize(self):
        """
        矩阵X的中心化

        返回:
        centerX (numpy.ndarray): 中心化的样本矩阵。
        """
        print("样本矩阵X:\n", self.X)
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 计算每个特征的均值
        centerX = self.X - mean  # 从每个样本中减去均值，实现中心化
        print("样本矩阵X的中心化centerX:\n", centerX)
        return centerX

    def _cov(self):
        """
        求样本矩阵X的协方差矩阵C

        返回:
        C (numpy.ndarray): 协方差矩阵。
        """
        ns = np.shape(self.centerX)[0]  # 样本数量
        C = np.dot(self.centerX.T, self.centerX) / (ns - 1)  # 计算协方差矩阵
        print("样本矩阵X的协方差矩阵C:\n", C)
        return C

    def _U(self):
        """
        求X的降维转换矩阵U，形状为 (n, k)，其中 n 是特征维度总数，k 是降维后的特征维度

        返回:
        U (numpy.ndarray): 降维转换矩阵。
        """
        a, b = np.linalg.eig(self.C)  # 计算协方差矩阵的特征值和特征向量
        print("X的协方差矩阵C的特征值a:\n", a)
        print("X的协方差矩阵C的特征向量b:\n", b)

        # 将特征值按降序排列，并选择前K个特征向量
        ind = np.argsort(-1 - a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print("样本矩阵X的降维转换矩阵U:\n", U)
        return U

    def _Z(self):
        """
        求样本矩阵X的降维矩阵Z

        返回:
        Z (numpy.ndarray): 降维后的样本矩阵。
        """
        Z = np.dot(self.X, self.U)  # 通过降维转换矩阵将样本矩阵降维
        print("X shape:\n", np.shape(self.X))
        print("U shape:\n", np.shape(self.U))
        print("Z shape:\n", np.shape(Z))
        print("Z:\n", Z)
        return Z

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
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1  # 降维后的特征数量为原特征数量减1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)

    # 使用CPCA类进行PCA降维
    pca = CPCA(X, K)
