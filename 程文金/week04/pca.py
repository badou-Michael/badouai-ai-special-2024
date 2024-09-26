import numpy as np

class CPCA(object):
    def __init__(self, X, K):
         self.X = X
         self.K = K
         self.centerX = []
         self.C = []   #样本矩阵X的协方差矩阵
         self.U = []   #样本矩阵X的降维转换矩阵
         self.Z = []   #样本矩阵X的降维矩阵

         self.centerX = self.centerized()
         self.C = self._cov()
         self.U = self._U()
         self.Z = self._Z()

    def centerized(self):
        print("输入的样本矩阵为：\n", self.X)
        mean = np.array([np.mean(attrs) for attrs in self.X.T])
        print("样本集的特征均值：\n", mean)
        centerX = self.X - mean
        print("样本集中心化后的矩阵：\n", centerX)
        return centerX

    def _cov(self):
        ns = np.shape(self.centerX)[0]
        C = np.dot(self.centerX, self.centerX.T)/(ns -1 )
        print("样本矩阵X的协方差矩阵\n", C)
        return C

    def _U(self):
        a, b  = np.linalg.eig(self.C)
        print("样本矩阵X的协方差矩阵的特征值：\n", a)
        print("样本矩阵X的协方差矩阵的特征值向量：\n", b)
        ind = np.argsort(-1* a)
        print("ind:\n", ind)
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print("%d样本矩阵X的降维矩阵\n"%self.K, U)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print("X shape:\n", self.X.shape)
        print("U shape:\n", self.U.shape)
        print("Z shape:\n", Z.shape)
        return Z


if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    K = np.shape(X)[1] - 1
    print("样本集为：\n", X)
    pca = CPCA(X, K)


