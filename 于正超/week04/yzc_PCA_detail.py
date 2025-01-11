"""
使用pca求样本矩阵X的k阶降维矩阵Z
"""
import numpy as np

class CPCA(object):
    """X为原样本矩阵，K为降维K阶矩阵
    先使用init 初始化变量"""

    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.centrX = []  #矩阵中心化
        self.C = []  #样本集协方差矩阵C
        self.U = [] ##样本矩阵X的降维转换矩阵
        self.Z = [] #样本矩阵X的降维矩阵Z，最终结果

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        """中心化"""
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        return centrX

    def _cov(self):
        """计算协方差矩阵C"""
        ns = np.shape(self.X)[0]
        C = np.dot(self.centrX.T,self.centrX)/(ns -1)
        print("协方差C：\n",C)
        return C

    def _U(self):
        """求X的降维转换矩阵U，"""
        a,b = np.linalg.eig(self.C)  #a为特征值，b为特征向量
        print("特征值：\n",a)
        print("特征向量： \n",b)
        ind = np.argsort(-1 * a)
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print("降维矩阵U： \n" , U)
        return U
    def _Z(self):
        """按照Z=XU 求降维矩阵Z"""
        Z = np.dot(self.X,self.U)
        print("样本矩阵X的降维矩阵Z：\n" , Z)
        return Z



if __name__ == '__main__':
    '10样本3特征；10行3列；行=样本，列=特征维度'
    X = np.array([[22,33,44],
                  [43,21,54],
                  [22,11,12],
                  [65,76,45],
                  [11,2,45],
                  [43,32,12],
                  [8,5,10],
                  [17,18,21],
                  [25,28,20],
                  [2 , 44, 31]])
    K = np.shape(X)[1] - 1
    pca = CPCA(X,K)
