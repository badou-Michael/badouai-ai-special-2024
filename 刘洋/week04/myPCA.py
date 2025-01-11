import numpy as np

class myPCA(object):
    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.centrX = []
        self.C = []   # 协方差矩阵C
        self.U = []   # 降维转换矩阵U
        self.Z = []   # 降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得
    def _centralized(self):
        # 样本数据中心化
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # np.mean求矩阵行的均值
        print('样本集的特征均值:\n', mean)  # （1，3）
        centrX = self.X - mean
        return centrX

    def _cov(self):
        # 求协方差矩阵C
        ns = np.shape(self.centrX)[0]  # 样本数量
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        # print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        # 求降维转换矩阵U(n,k), n是X的特征维度，k是降维矩阵的特征维度
        a, b = np.linalg.eig(self.C)   # 特征值赋值给a，对应特征向量赋值给b(b的列向量对应特征向量)
        # 找特征值降序排列的索引，找前K大的特征值
        ind = np.argsort(-1*a)   # [0 1 2]
        # 构建K阶降维的降维转换矩阵U
        UT = []
        for i in range(self.K):
            UT.append(b[:,ind[i]])
        U = np.transpose(UT)
        # print('%d阶降维转换矩阵U:\n'%self.K, U)
        return U

    def _Z(self):
        # 求降维矩阵Z(m,k)
        Z = np.dot(self.X, self.U)  # （10，3）*（3，2）
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
    pca = myPCA(X, K)
    print('样本矩阵X的降维矩阵Z:\n', pca.Z)

