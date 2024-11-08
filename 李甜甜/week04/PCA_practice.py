import numpy as np
class CPCA(object):
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centerX = []
        self.C = []
        self.U = []
        self.z = []

        self.centerX = self._centralized()
        self.C =self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        print('样本矩阵X:\n',self.X)
        centerX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('样本集的特征均值：\n',mean)
        centerX = self.X - mean
        print('样本矩阵X的中心化X:\n', centerX)
        return centerX

    def _cov(self):
        ns = np.shape(self.centerX)[0]
        C = np.dot(self.centerX.T,self.centerX)/(ns-1)
        print('样本矩阵x的协方差矩阵c：\n', C)
        return C

    def _U(self):
        a,b = np.linalg.eig(self.C)
        print('样本集协方差矩阵c的特征值：\n', a)
        print('样本集协方差矩阵c的特征向量：\n', b)
        ind = np.argsort(-1*a)
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n'%self.K, U)
        return U
    def _Z(self):
        Z =np.dot(self.X,self.U)
        print ('X shape:',np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

if __name__=='__main__':
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
    pca = CPCA(X, K)


