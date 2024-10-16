import numpy as np
#使用PCA求样本矩阵X的K阶降维矩阵Z
"""
1.数据中心化 - 中心化后的数据计算协方差矩阵时会更加方便，能够更好地突出数据的主要特征。
X = X - np.mean(X, axis=0)  axis=0表示计算每一列的均值，如果是一维数组则返回整个数组的均值
2.计算协方差矩阵
C = np.dot(X.T, X) / (N - 1)  N是样本的个数
C = np.cov(X.T,rowvar=False)  
3.计算协方差矩阵的特征值和特征向量

"""



class Principal_ConponentA_analysis(object):
    def __init__(self, X, K):
        """
         :param X:样本矩阵
         :param K:样本矩阵的降维矩阵的阶数
         :return:
        """
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化后的矩阵
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized() # 样本中心化矩阵
        self.C = self._cov() # 样本协方差矩阵
        self.U = self._U() # 样本矩阵X的降维转换矩阵
        self.Z = self._Z()  # Z=XU求得
    def _centralized(self):
        """
        样本中心化
        """
        print("样本矩阵X：\n", self.X)
        centrX = self.X - np.mean(self.X, axis=0)
        print("样本矩阵X的中心化：\n", self.centrX)
        return centrX

    def _centralized1(self):
        """
        样本中心化方法2
        """
        print("样本矩阵X：\n", self.X)
        #计算样本均值
        #self.X.T 是将样本专制，有M*N变成N*M 这样可以计算出每一列的均值
        mean = np.array([np.mean(attr) for attr in self.X.T]) #np.mean(attr)计算每一列的均值
        print("样本矩阵X的均值：\n", mean)
        #计算样本中心化矩阵
        centrX = self.X - mean
        print("样本矩阵X的中心化：\n", self.centrX)
        return centrX
    def _cov(self): #计算样本矩阵的协方差矩阵
        """
        样本协方差矩阵
        """
        C = np.cov(self.centrX, rowvar=False)  # rowvar=False表示每一列代表一个样本
        print("样本矩阵X的协方差矩阵：\n", self.C)
        return C
    def _cov1(self):
        """
        样本协方差矩阵方法2
        """
        #获取样本样例总数
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print("样本矩阵X的协方差矩阵：\n", C)
        return C
    def _U(self): #计算样本矩阵的降维转换矩阵
        """
        样本矩阵X的降维转换矩阵
        eigVals 是特征值
        eigVects 是特征向量
        """
        eigVals, eigVects = np.linalg.eig(self.C)  # 计算特征值和特征向量
        print("样本矩阵X的特征值：\n", eigVals)
        print("样本矩阵X的特征向量：\n", eigVects)
        # 对特征值从大到小降序的索引 eg:eigValIndice[0]为最大值
        eigValIndice = np.argsort(-1*eigVals) # [0 1 2]
        # 构建K阶降维的降维转换矩阵U,矩阵的切片array[a:b,c:d]表示从a到b行，c到d列 a:b表示行，为空表示取所有行
        # 一下代码表示取特征向量的所有行，前俩列 这里K为2
        #这里UT返回了一个俩行三列的数组,取了特征值最大的俩列
        UT = [eigVects[:,eigValIndice[i]] for i in range(self.K)]
        print("======UT======\n", UT)
        U = np.transpose(UT) #转置操作为三行俩列
        print("样本矩阵X的降维转换矩阵：\n", U)
        return U
    def _Z(self): #计算样本矩阵的降维矩阵
        #降维矩阵Z的算法 Z=XU，X为原始样本，U为降维转换矩阵
        #np.dot(x,y) 表示矩阵x和y相乘
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z
if __name__ == '__main__':
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
    print("样本集为10行3列，10个样本，每个样本3个特征\n", X)
    pca = Principal_ConponentA_analysis(X, K)


