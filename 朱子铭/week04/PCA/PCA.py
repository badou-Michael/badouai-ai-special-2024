# -*- coding: utf-8 -*-
"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""

import numpy as np

# 定义一个名为 CPCA 的类，用于执行主成分分析（PCA）操作
class CPCA(object):
    def __init__(self, X, K):
        '''
        :param X,样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        # 初始化输入样本矩阵 X 和降维阶数 K
        self.X = X
        self.K = K
        # 初始化属性，分别用于存储中心化后的样本矩阵、协方差矩阵、降维转换矩阵和降维后的矩阵
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.T = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        # 调用类中的私有方法计算并赋值各个属性
        self.centrX = self._centralized()
        self.C = self._COV()
        self.T = self._T()
        self.Z = self._Z()

    # 私有方法，用于计算样本矩阵的中心化
    def _centralized(self):
        # 打印原始样本矩阵，方便查看输入数据
        print("样本矩阵:\n",self.X)
        # 计算样本矩阵在每个特征维度上的均值
        # 参数 axis=0 表示沿着列的方向计算均值，即计算每个特征的均值
        mean = np.mean(self.X, axis=0)
        # 打印样本集的特征均值，方便了解数据的整体特征
        print('样本集的特征均值:\n', mean)
        # 计算样本矩阵的中心化，即将每个样本减去特征均值
        centrX = self.X - mean
        # 打印中心化后的样本矩阵，查看中心化后的结果
        print('样本矩阵X的中心化centrX:\n', centrX)
        # 返回中心化后的样本矩阵，供后续方法使用
        return centrX

    # 私有方法，用于计算样本集的协方差矩阵
    def _COV(self):
        # 获取中心化后的样本矩阵的行数，即样本数量
        ns = self.centrX.shape[0]
        # 计算协方差矩阵
        # np.dot(self.centrX.T,self.centrX) 计算中心化后矩阵的转置与自身的乘积
        # 除以 (ns - 1) 是为了得到样本协方差矩阵的无偏估计
        C = np.dot(self.centrX.T,self.centrX)/(ns-1)
        # 打印协方差矩阵，方便查看计算结果
        print("协方差矩阵C：\n",C)
        # 返回协方差矩阵，供后续方法使用
        return C

    # 私有方法，用于计算样本矩阵的降维转换矩阵
    def _T(self):
        # 计算协方差矩阵的特征值和特征向量
        # np.linalg.eig 函数用于计算方阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(self.C)
        # 打印协方差矩阵的特征值，了解数据在不同特征方向上的方差大小
        print('样本集的协方差矩阵C的特征值:\n', eigenvalues)
        # 打印协方差矩阵的特征向量，了解数据在不同特征方向上的变化方向
        print('样本集的协方差矩阵C的特征向量:\n', eigenvectors)
        # 对特征值进行排序，获取从大到小的索引
        # eigenvalues.argsort() 返回特征值从小到大的索引
        # [::-1] 将索引反转，得到从大到小的索引
        sort_indices = eigenvalues.argsort()[::-1]
        # 根据索引选取前 K 个特征向量组成降维转换矩阵
        # eigenvectors[:, sort_indices[:self.K]] 选取排序后的前 K 个特征向量
        top_k_eigenvectors = eigenvectors[:, sort_indices[:self.K]]
        # 打印状态转移矩阵，即降维转换矩阵，查看降维的方向选择
        print("状态转移矩阵：\n", top_k_eigenvectors)
        # 返回降维转换矩阵，供后续方法使用
        return top_k_eigenvectors

    # 私有方法，用于计算样本矩阵的降维矩阵
    def _Z(self):
        # 计算降维后的矩阵
        # np.dot(self.X,self.T) 用原始样本矩阵乘以降维转换矩阵，实现降维操作
        Z = np.dot(self.X,self.T)
        # 打印降维后的矩阵，查看降维结果
        print('样本矩阵X的降维矩阵Z:\n', Z)
        # 返回降维后的矩阵，作为该方法的输出结果
        return Z

# 主程序入口
if __name__=='__main__':
    # 定义一个 10 样本 3 特征的样本集，行为样例，列为特征维度
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14]])
    # 计算降维阶数 K，这里取样本矩阵的列数减 1
    K = np.shape(X)[1] - 1
    # 打印原始样本矩阵，方便查看输入数据
    print('样本集(10 行 3 列，10 个样例，每个样例 3 个特征):\n', X)
    # 创建 CPCA 类的实例，传入样本矩阵 X 和降维阶数 K，进行主成分分析
    pca = CPCA(X,K)
