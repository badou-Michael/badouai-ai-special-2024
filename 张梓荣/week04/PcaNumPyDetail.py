import numpy as np


class CPCA(object):
    def __init__(self, X, K):
        self.X = X
        self.K = K
        # 中心化矩阵
        self.central = self._centralized()
        # 协方差矩阵
        self.cov = self._cov()
        # 降维转换矩阵(特征向量组成)
        self.charters = self._charters()
        # 目标矩阵
        self.target = self._target()

    # 得到中心化矩阵
    def _centralized(self):
        means = np.array([np.mean(attr) for attr in self.X.T])
        return self.X - means

    # 得到协方差矩阵
    def _cov(self):
        return np.dot(self.central.T, self.central) / (self.central.shape[0] - 1)

    # 得到目标特征向量的组合体
    def _charters(self):
        a, b = np.linalg.eig(self.cov)
        return np.transpose(np.array([b[:, np.argsort(-1 * a)[i]] for i in range(self.K)]))

    # 得到目标矩阵(降维后的)
    def _target(self):
        return np.dot(self.X, self.charters)


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
    K = X.shape[1] - 1
    pca = CPCA(X, K)
    print(pca.target)