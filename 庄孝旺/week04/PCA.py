import numpy as np


class CPCA(object):

    def __init__(self, X, K):
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []
        self.Z = []
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        return centrX

    def _cov(self):
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        return C

    def _U(self):
        a, b = np.linalg.eig(self.C)
        ind = np.argsort(-1 * a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        return Z


if __name__ == '__main__':
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
    pca = CPCA(X, K)
