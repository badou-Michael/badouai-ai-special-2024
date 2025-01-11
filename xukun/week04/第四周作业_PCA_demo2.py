import numpy as np
import random


# 使用PCA降维
class PCAA(object):
    '''

    :param X: 样本集
    :param k: 降维后的维度
    :return:
    '''

    def __init__(self, X, K):
        self.X = X
        self.k = K
        self.centerX = []  # 中心化数据
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centerX = self.centerXs()
        self.C = self.cov()
        self.U = self.eig()
        self.Z = self.transform()

    def centerXs(self):
        print("样本矩阵X...\n", self.X)
        centerX = np.mean(self.X, axis=0)
        print("centerX特征值...\n", centerX)
        self.centerX = self.X - centerX
        print("中心化后的数据...\n", self.centerX)
        return self.centerX

    def cov(self):
        C = np.dot(self.centerX.T, self.centerX) / (self.centerX.shape[0] - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def eig(self):
        eig_vals, eig_vecs = np.linalg.eig(self.C)
        print('样本矩阵X的特征值:\n', eig_vals)  # 打印特征值
        print('特征向量是:\n', eig_vecs)  # 打印特征向量
        idx = np.argsort(-eig_vals)
        UT = eig_vecs[:, idx[:self.k]]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.k, U)
        return U

    def transform(self):
        Z = np.dot(self.X, self.U)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    X = np.array([[1, 21, 35],
                  [2, 22, 34],
                  [3, 23, 33],
                  [4, 24, 32],
                  [5, 25, 31],
                  [10, 29, 39],
                  [11, 28, 38],
                  [12, 27, 37],
                  [13, 26, 36],
                  [14, 25, 35]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCAA(X, K)
