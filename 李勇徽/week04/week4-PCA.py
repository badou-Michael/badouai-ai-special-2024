import numpy as np
import random
from sklearn.decomposition import PCA

class CPCA(object):
    '''
    使用PCA获得样本矩阵X的K阶降维矩阵Z
    '''
    def __init__(self, X, K):
        '''
        param X: 样本矩阵X
        param K: X的降维矩阵阶数
        '''
        self.X = X       # 样本矩阵X
        self.K = K       # 降维矩阵的阶数
        self.centrX = [] # 矩阵X的中心化
        self.C = []      # 样本集的协方差矩阵C
        self.U = []      # 样本矩阵X的降维转换矩阵
        self.Z = []      # 样本矩阵X的降维矩阵Z
        
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z() #Z=XU求得

    def _centralized(self):
        '''
        对样本矩阵X进行零均值化
        '''
        centrX = []
        # 求各维度均值
        mean = np.array([np.mean(arr) for arr in self.X.T])
        # 零均值化
        centrX = self.X - mean
        return centrX
    def _cov(self):
        '''
        根据中心化后的矩阵，求协方差矩阵
        '''
        # n为样本个数
        n = np.shape(self.centrX)[0]
        # 求协方差矩阵
        cov = np.dot(self.centrX.T, self.centrX) / (n-1)
        return cov
    def _U(self):
        '''
        根据协方差矩阵，求K阶降维转换矩阵U
        '''
        # 求协方差矩阵的特征值和特征向量，分别赋值给a和b
        a, b = np.linalg.eig(self.C)
        # 根据得到的特征值进行排序，得到TopK的索引
        ind = np.argsort(-1*a)
        # 根据索引获得对应特征向量
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U
    def _Z(self):
        '''
        获得降维转换矩阵U后，可以将样本矩阵X投射到选择的特征向量U上，得到降阶矩阵Z
        '''
        # 将样本矩阵投射到K阶降维转换矩阵U
        Z = np.dot(self.X, self.U)
        print("method1: 样本矩阵X的降维矩阵Z:\n", Z)
        return Z

def get_random_array(x=10, y=3):
    '''
    param x: 生成样本数据样例数
    param y: 生成样本数据维度
    '''
    res = []
    for i in range(x):
        tmp_row = []
        for j in range(y):
            tmp_row.append(random.randint(1,100))
        res.append(tmp_row)
    return res

def PCA_sklearn(X, K):
    '''
    param X: 样本矩阵
    param K: 降维阶数
    '''
    # 降到K维
    pca = PCA(n_components=K)
    # 执行
    pca.fit(X)
    Z = pca.fit_transform(X)
    print("method2: 样本矩阵X得降维矩阵Z:\n", Z)
    return Z


if __name__ == '__main__':
    # X为样本集
    sample = get_random_array(10, 3)
    X = np.array(sample)

    # K为降维
    K = np.shape(X)[1] - 1

    # 方法一：手动实现
    pca = CPCA(X, K)

    # 方法二：调用接口
    # pca = PCA_sklearn(X, K)

    