import numpy as np
import cv2
import random

class CPCA(object):
    def __init__(self,X,K):         # 以_开头的方法通常被视为私有方法
        self.X = X      # 存储样本矩阵
        self.K = K      # 存储降维矩阵的维数，k阶
        self.centerX = []   # 存储矩阵中心化后的结果
        self.C = []     # 存储协方差矩阵
        self.U = []     # 存储样本矩阵X的降维转换矩阵
        self.Z = []     # 存储样本矩阵X的降维结果

        self.centerX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    # 对矩阵X进行中心化
    def _centralized(self):
        print('样本矩阵为：\n',self.X)
        centraX = []        # 初始化一个空列表，存储中心化后的数据
        mean = np.array([np.mean(attr) for attr in self.X.T])
        # 计算样本集每个特征的均值,
        # 通过 self.X.T获取样本矩阵X的转置（即各列代表不同的特征）,然后计算每个特征的均值,并将这些均值存储在 mean 数组中
        print('样本集的特征均值：\n',mean)

        centraX = self.X - mean     # 对样本矩阵 X 进行中心化操作(从每个特征值中减去其均值，使得每个特征的平均值为零)
        print('中心化后的样本矩阵：\n',centraX)

        return centraX

    # 求样本矩阵X的协方差矩阵C
    def _cov(self):
        exampleSum = np.shape(self.centerX)[0]
        # 获取中心化后样本矩阵的行数（即样例的总数）
        # np.shape(self.centrX)返回centrX的形状（维度），[0]获取第一维的大小，即样例的总数。

        C = np.dot(self.centerX.T,self.centerX)/(exampleSum -1)
        # 计算协方差矩阵。这里使用了NumPy的 dot 函数来计算矩阵的乘积
        # self.centrX.T是中心化后样本矩阵的转置（即特征的转置），用于计算不同特征之间的协方差，并除以 (ns - 1)（样本数减一）来进行无偏估计。

        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    # 求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度
    def _U(self):
        a,b = np.linalg.eig(self.C)
        # 利用linalg.eig函数计算协方差矩阵 self.C的特征值和特征向量。特征值赋值给 a，对应的特征向量赋值给 b
        print('样本的协方差矩阵的特征值：\n',a)
        print('样本的协方差矩阵的特征向量：\n',b)

        DESC_a = np.argsort(-1 *a)    # 使用np.argsort函数对特征值a进行降序排序,找出最大的self.K个特征值对应的特征向量
        UT = [b[:,DESC_a[i]] for i in range(self.K)]
        # b[:, ind[i]] 表示从特征向量矩阵b中提取第 DESC_a[i] 列的特征向量
        U = np.transpose(UT)    # 使用transpose函数，将UT转置
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    # 按照Z=XU求降维矩阵Z
    def _Z(self):
        Z = np.dot(self.X,self.U)
        # 使用NumPy的dot函数计算矩阵乘法，将原始样本矩阵X和降维转换矩阵 U 相乘，得到降维后的矩阵 Z
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
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
    K = np.shape(X)[1] - 1
    # 第一维度（样本数）：10，表示10个样本。
    # 第二维度（特征数）：3，表示每个样本有3个特征。
    # 通常PCA的转换矩阵不包含原始特征的均值项，经过中心化处理后，特征的均值为0，所以要-1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X,K)