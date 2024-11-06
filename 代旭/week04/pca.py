import numpy as np

class CPCA(object):
    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.centrX = []
        self.C=[]
        self.U=[]
        self.Z=[]

        self.centrX = self._centralized()


    def _centralized(self):
        print('样本集：',self.X)
        centrX=[]
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('样本集的特征均值:',mean)
        centrX=self.X-mean
        print('样本矩阵X的中心化',centrX)
        return centrX

    def _cov(self):
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T,self.centrX)/(ns-1)
        print('样本矩阵X的协方差矩阵C：',C)
        return C

    def _U(self):
        a,b=np.linalg.eig(self.C)
        print('特征值：',a)
        print('特征向量：',b)
        ind = np.argsort(-1*a)
        UT = [b[:ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d降维转换矩阵U：'%self.K,U)
        return U

    def _Z(self):
        Z=np.dot(self.X,self.U)
        print('X shape:',np.shape(self.X))
        print('U shape:',np.shape(self.U))
        print('Z shape:',np.shape(self.Z))
        return Z

if __name__=='__main__':
    X=np.array([[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 5],
                  [4, 5, 6],
                  [5, 6, 7],
                  [6, 7, 8],
                  [7, 8, 9],
                  [8, 9, 10],
                  [9, 10, 11],
                  [10, 11, 12]])
    K=np.shape(X)[1]-1
    print('样本集（10行3列，10个样例，每个样例3个特征）：',X)
    pca = CPCA(X,K)
