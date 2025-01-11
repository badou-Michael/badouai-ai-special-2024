"""
手写pca细节
"""

import numpy as np


class PCA(object):
    def __init__(self, X, K):  # 参数X为样本矩阵，参数K为降维值，降到多少维度
        self.X = X  # 样本矩阵
        self.K = K
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵
        self.U = []  # 样本集的降维转换矩阵
        self.Z = []  # 样本矩阵降维之后的矩阵

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        # 矩阵中心化
        print("样本矩阵:\n", self.X)
        centrX = []
        # 采用列表推导式的方式去计算列均值
        # mean = np.array([np.mean(attr) for attr in self.X.T])
        # 采用接口方式
        mean = np.mean(self.X, axis=0)
        print("样本均值:\n", mean)
        centrX = self.X - mean
        print("样本中心化矩阵:\n", centrX)
        return centrX

    def _cov(self):
        # 求协方差矩阵，先取出样本个数（多少列就是多少个样本）
        num = np.shape(self.centrX)[0]
        print("样本的个数：\n", num)
        C = np.dot(self.centrX.T, self.centrX) / (num - 1)
        return C

    def _U(self):
        # 求X的降维转换矩阵U,shape=(n,k).n是特征维度，k是降维的数
        # 先求协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(
            self.C)  # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出索引，按照从大到小排列，取出比较大的特征值
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        print('UT:\n', UT)
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        # 按照Z = XU公式来求降维后的矩阵
        Z = np.dot(self.centrX, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X, K)
