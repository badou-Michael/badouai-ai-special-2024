# -*- coding: utf-8 -*-
"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""

import numpy as np


class PCA(object):
    '''
    使用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''

    def __init__(self, x, k):
        """
        :param x: 样本矩阵X
        :param k: X的降维矩阵的阶数，即X要特征降维成k阶
        """
        self.sample_matrix = x  # 样本矩阵X
        self.dimensionality_reduction_order = k  # K阶降维矩阵的K值
        self.centered_matrix = []  # 矩阵X的中心化
        self.covariance_matrix = []  # 样本集的协方差矩阵C
        self.dimensionality_reduction_matrix = []  # 样本矩阵X的降维转换矩阵
        self.dimensionality_reduced_matrix = []  # 样本矩阵X的降维矩阵Z

        self.centered_matrix = self._center_samples()
        self.covariance_matrix = self._calculate_covariance()
        self.dimensionality_reduction_matrix = self._calculate_dimensionality_reduction_matrix()
        self.dimensionality_reduced_matrix = self._calculate_dimensionality_reduced_matrix()  # Z=XU求得

    def _center_samples(self):
        '''矩阵X的中心化'''
        print('样本矩阵X:\n', self.sample_matrix)
        centered_matrix = []
        mean = np.array([np.mean(attr) for attr in self.sample_matrix.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        centered_matrix = self.sample_matrix - mean  ##样本集的中心化
        print('样本矩阵X的中心化centered_matrix:\n', centered_matrix)
        return centered_matrix

    def _calculate_covariance(self):
        '''求样本矩阵X的协方差矩阵C'''
        # 样本集的样例总数
        number_of_samples = np.shape(self.centered_matrix)[0]
        # 样本矩阵的协方差矩阵C
        covariance_matrix = np.dot(self.centered_matrix.T, self.centered_matrix) / (number_of_samples - 1)
        print('样本矩阵X的协方差矩阵C:\n', covariance_matrix)
        return covariance_matrix

    def _calculate_dimensionality_reduction_matrix(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        # 先求X的协方差矩阵C的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(
            self.covariance_matrix)  # 特征值赋值给eigenvalues，对应特征向量赋值给eigenvectors。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', eigenvalues)
        print('样本集的协方差矩阵C的特征向量:\n', eigenvectors)
        # 给出特征值降序的topK的索引序列
        indices = np.argsort(-1 * eigenvalues)
        # 构建K阶降维的降维转换矩阵U
        U_transpose = [eigenvectors[:, indices[i]] for i in range(self.dimensionality_reduction_order)]
        dimensionality_reduction_matrix = np.transpose(U_transpose)
        print('%d阶降维转换矩阵U:\n' % self.dimensionality_reduction_order, dimensionality_reduction_matrix)
        return dimensionality_reduction_matrix

    def _calculate_dimensionality_reduced_matrix(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        dimensionality_reduced_matrix = np.dot(self.sample_matrix, self.dimensionality_reduction_matrix)
        print('X shape:', np.shape(self.sample_matrix))
        print('U shape:', np.shape(self.dimensionality_reduction_matrix))
        print('Z shape:', np.shape(dimensionality_reduced_matrix))
        print('样本矩阵X的降维矩阵Z:\n', dimensionality_reduced_matrix)
        return dimensionality_reduced_matrix


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    sample_matrix = np.array([[10, 15, 29],
                              [15, 46, 13],
                              [23, 21, 30],
                              [11, 9, 35],
                              [42, 45, 11],
                              [9, 48, 5],
                              [11, 21, 14],
                              [8, 5, 15],
                              [11, 12, 21],
                              [21, 20, 25]])
    dimensionality_reduction_order = np.shape(sample_matrix)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', sample_matrix)
    pca = PCA(sample_matrix, dimensionality_reduction_order)
