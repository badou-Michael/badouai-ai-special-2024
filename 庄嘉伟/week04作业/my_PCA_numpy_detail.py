#PCA手写版
#PCA主成分分析是指将一个三维特征变为二维特征的方法
import numpy as np



class CPCA(object):
    # 用PCA求样本矩阵X的K阶降维矩阵
    #请保证输入的样本矩阵X shape = （m，n）
    def __init__(self,X,K):
        """X表示要降维的矩阵，K表示降低后的阶数"""
        self.X = X
        self.K = K
        self.centrX = [] #矩阵X的中心化
        self.C = [] #样本的协方差矩阵C
        self.U = [] #样本矩阵X的降维转换矩阵
        self.Z = [] #样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()#矩阵X的中心化（中心化为所有数据减均值）
        self.C = self._cov()  #求协方差矩阵
        self.U = self._U()  #降维矩阵，求特征值特征向量
        self.Z = self._Z()  #Z = XU

    def _centralized(self):
        print('样本矩阵:X\n',self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])#样本集特征值的平均值，T为转置
        print("样本特征平均值:\n",mean)
        centrX = self.X - mean #去中心化
        print("样本矩阵X的中心化centrX:\n",centrX)
        return centrX

    def _cov(self):
        #求样本X的协方差矩阵C
        ns = np.shape(self.centrX)[0]         #样本集的样本总数
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)  #求协方差矩阵公式
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        #求X的降维转换矩阵U，shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度
        #先求X的协方差矩阵C的特征值和特征向量
        a,b = np.linalg.eig(self.C) #特征值赋值给a，对应特征向量赋值给b。调用接口算特征值和特征向量
        # 函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        #给出特征值降序的topk的索引序列
        ind = np.argsort(-1*a)  #变成从大到小排序
        #构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        #展开写法：
        """
            UT = []  # 初始化空列表

            for i in range(self.K):  # 遍历从 0 到 self.K - 1 的索引
                column = b[:, ind[i]]  # 提取 b 中的第 ind[i] 列
                UT.append(column)  # 将提取的列添加到 UT 列表中
        """
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n'%self.K, U)
        return U
    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

if __name__ =="__main__":
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
    K = np.shape(X)[1]-1   #np.shape(X)表示返回X的行列数。K为X的列数-1
    print("样本集（10行3列，10个样本，每个样本三个特征）:\n",X)
    pca = CPCA(X,K)




