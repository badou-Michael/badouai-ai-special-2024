
import numpy as np

class PCA(object):

    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.centrX = []
        self.C = []
        self.U = []
        self.Z = []

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        mean = np.array([np.mean(a) for a in self.X.T])
        centrX = self.X-mean
        return centrX

    def _cov(self):
        C = np.dot(self.centrX.T,self.centrX)/(np.shape(self.centrX)[0]-1)
        return C

    def _U(self):
        a,b = np.linalg.eig(self.C)
        d = np.argsort(-a)
        U = b[:,d[:self.K]]
        return U
    def _Z(self):
        Z = np.dot(self.X,self.U)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

if __name__=='__main__':
    X = np.array([[10, 15, 29,23],
                  [42, 45, 11, 23],
                  [9,  48, 5, 13],
                  [11, 21, 14, 34],
                  [15, 46, 13, 56],
                  [23, 21, 30, 37],
                  [11, 9, 35, 9],
                  [8,  5,  15, 45],
                  [11, 12, 21, 20],
                  [21, 20, 25, 23]])
    pca = PCA(X, 3)
