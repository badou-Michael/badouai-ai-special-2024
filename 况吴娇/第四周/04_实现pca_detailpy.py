# PCA（主成分分析）是一种常用的数据降维技术，它通过线性变换将原始数据映射到一个新的坐标系中，
# 新坐标系的选择是使得数据在新坐标系下的方差最大化
#PCA：Principal Component Analysis 主成分分析

import  numpy as np
class CPCA(object):
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''
    def __init__(self, X, K):
        '''
        :param X,样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.X=X
        self.K=K
        self.centrX=[]
        self.C =[]
        self.U = []
        self.Z = []

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得
    def _centralized(self):
        '''矩阵X的中心化'''
        print('样本矩阵X:\n', self.X)
        centrX = []
        #np.array  用于创建一个 NumPy 数组
        mean=np.array([np.mean(attr) for attr in self.X.T ]) ##中心化的目的是将每个特征的均值变为0，而转置是为了计算每个特征的均值(K)
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  # 样本集的中心化
        print('样本矩阵X的中心化:\n',centrX)
        return centrX
    def _cov(self):##求协方差--样本总数，样本矩阵X的协方差矩阵C
        ns = np.shape(self.centrX)[0]
        ##样本协方差矩阵
        C = np.dot(self.centrX.T, self.centrX)/ (ns - 1) #分母使用  ns-1  是为了得到一个无偏估计的样本方差和协方差，这是统计学中的一个标准做法。
        #协方差矩阵的计算，顺序确实很重要 结果是一个 (n×n) 的方阵，表示特征间的协方差。
        print('样本矩阵X的协方差矩阵C\n',C)
        return C
    def _U(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
    # 先求X的协方差矩阵C的特征值和特征向量
    #函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
    # 这一步是计算协方差矩阵的特征值和特征向量，然后选择最大的K个特征值对应的特征向量。这些特征向量构成了降维转换矩阵。
        a,b=np.linalg.eig(self.C) #特征值赋值给a，对应特征向量赋值给b。
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
    #把特征值作为索引，然后降序，
        ind=np.argsort(-1*a)
    #然后用特征向量b, 选择最大的K个特征值对应的特征向量 # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]]for i in range(self.K)]   #在Python的索引中，冒号 : 表示选择所有行，而 ind[i] 表示选择第 i 个特征向量。
    #    对于K个最大的特征值（按索引ind排序），选择 b矩阵中对应的列（即特征向量），将这些列（特征向量）存储在列表UT 中。
        U=np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U
    #在PCA中，我们通常希望降维转换矩阵 U 的每一行代表一个特征向量，这样在后续计算中，原始数据矩阵 X 可以通过与 U 相乘来进行降维。
    #构造降维转换矩阵：np.transpose(UT) 将 UT 列表中的每个特征向量（原本是列向量）转置成行向量，并将这些行向量组合成一个矩阵 U。这样，U 的每一行都对应于一个主成分，方便进行矩阵乘法操作
    def _Z(self):
        ''' 按照Z = XU求降维矩阵Z, shape = (m, k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
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
    K= np.shape(X)[1] - 1 #K=X.shape[1]-1
    print ('样本集为10行3列，10个样例（行）,每个样例3特称，\n X',X)
    pca = CPCA(X, K)







# 1. 导入 NumPy 库
# 2. 定义 CPCA 类
# 类接受样本矩阵  X  和降维目标  K ，然后依次执行中心化、计算协方差矩阵、计算降维转换矩阵和计算降维矩阵。
# 中心化是将每个特征的均值变为0的过程。这是通过从每个特征中减去其均值来实现的。
# 计算协方差矩阵
# 协方差矩阵描述了数据特征之间的关系。在这个实现中，协方差矩阵是通过将中心化后的矩阵与其转置相乘，然后除以样本数减1来计算的。
# 计算降维转换矩阵
# 这一步是计算协方差矩阵的特征值和特征向量，然后选择最大的K个特征值对应的特征向量。这些特征向量构成了降维转换矩阵。
# 计算降维矩阵
# 降维矩阵是通过将原始数据矩阵  X  与降维转换矩阵  U  相乘得到的。(计算涉及点积)
# 7. 主程序if __name__=='__main__':
# 在主程序中，定义了一个样本矩阵  X  和一个降维目标  K ，然后创建了  CPCA  类的实例，执行 PCA 过程。
