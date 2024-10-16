import numpy as np

class CPCA(object):

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
        print('矩阵样本X：\n',self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        print('样本矩阵中心化：',centrX)
        return centrX
    def _cov(self):
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T,self.centrX)/(ns-1)
        print('样本矩阵的协方差矩阵：',C)
        return C
    def _U(self):
        a,b = np.linalg.eig(self.C)         #求特征值特征向量函数
        print('协方差矩阵的特征值：',a)
        print('协方差矩阵的特征向量：',b)
        ind = np.argsort(-1*a)              #argsort是升序排列，所以*-1
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U：\n'%self.K,U)
        return U
    def _Z(self):
        Z = np.dot(self.X,self.U)
        print('样本矩阵的降维矩阵：\n',Z)
        return Z




if __name__ == '__main__':
    X = np.array([[20,2,33],
                 [12,23,45],
                 [22,34,66],
                  [32,66,45],
                  [23,65,43],
                  [2,2,5],
                  [33,24,67],
                  [21,64,23],
                  [33,55,87],
                  [90,98,36]
                  ])
    K = np.shape(X)[1]-1
    print('样本集10行3列：\n',X)
    pca = CPCA(X,K)
