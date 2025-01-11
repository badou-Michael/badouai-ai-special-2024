import numpy as np
import matplotlib.pyplot as plt


# 自定义的PCA详细算法

class MPCA():
    def __init__(self, X, k):
        self.X = X  # n行样本 m列特征
        self.k = k  # m个特征要降成k个

        # 实例变量
        self.X_c = []  # 样本矩阵X的 中心化矩阵 X_centralise
        self.C = []  # 样本矩阵X的特征列向量的 协方差矩阵 Covariance
        self.eigen_vals = []  # 协方差矩阵的 特征值 eigen_value
        self.eigen_vecs = []  # 协方差矩阵的 特征向量 eigen_vectors
        self.W = []  # 样本矩阵X的 降维转换矩阵/主成分矩阵 W
        self.Y = []  # 样本矩阵X[不是X_c]降维后得到的 新样本矩阵/投影矩阵 Y=XW

        # 计算步骤
        self._centralise()  # 样本矩阵的中心化矩阵
        self._covariance()  # 特征向量的协方差矩阵
        self._transformation()  # 协方差矩阵的特征值和特征向量，样本矩阵的降维转换矩阵
        self._projected_data()  # 样本矩阵降维后的新样本矩阵

    def _centralise(self):
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 语法
        print('样本矩阵的特征均值矩阵\n', mean)
        self.X_c = X - mean  # 语法
        print('样本矩阵的特征中心化矩阵\n', self.X_c)

    def _covariance(self):
        n = X.shape[0]
        self.C = np.dot(self.X_c.T, self.X_c) / (n - 1)
        print('样本矩阵的协方差矩阵\n', self.C)

    def _transformation(self):
        self.eigen_vals, self.eigen_vecs = np.linalg.eig(self.C)
        print('协方差矩阵的特征值\n', self.eigen_vals)
        print('协方差矩阵的特征向量\n', self.eigen_vecs)
        sort_ind = np.argsort(-1 * self.eigen_vals)
        self.W = self.eigen_vecs[:, sort_ind[:self.k]]
        print('样本矩阵的降维转换矩阵\n', self.W)  # 语法

    def _projected_data(self):
        self.Y = np.dot(self.X, self.W)
        print('样本矩阵的投影矩阵\n', self.Y)


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
k = X.shape[1] - 1
print('样本矩阵\n', X)
print('样本矩阵的特征维度：{}，样本矩阵的目标降维特征维度k：{} '.format(X.shape[1], k))
mpca = MPCA(X, k)
# Y的散点图
x0, y0 = mpca.Y.T[0], mpca.Y.T[1]
x, y = x0.T, y0.T
print('x\n',x,'\ny\n',y)
plt.scatter(x,y,c='g',marker='.')
plt.show()
