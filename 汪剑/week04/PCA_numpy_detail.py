import numpy as np

'''
PCA（主成分分析）：是一种常见的降维算法。
通过将高维的数据映射到低维空间，同时保留原数据的主要特征，从而减少数据的维度，去除噪声，提升计算效率

实现PCA的一般步骤：
1、对原始数据零均值化（中心化）
2、求协方差矩阵
3、对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间
'''


class CPCA(object):
    '''
    输入的样本矩阵X shape(m,n)，m行样例，n个特征
    行数 = 样本数
    列数 = 特征数（变量数）

    X：输入的样本矩阵X
    K：X的降维矩阵的阶数，K即表示矩阵X需要降维成K阶矩阵
    '''

    # 初始化
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centerX = []  # 定义矩阵X的中心化矩阵，即：矩阵X 减去 每一列的特征（变量）的均值形成的矩阵后的矩阵
        self.C = []  # 定义样本集对应的协方差矩阵C，原矩阵X对应的有n维（列数）则得到的协方差矩阵是 nxn矩阵
        self.U = []  # 定义样本矩阵X降维的转换矩阵U
        self.Z = []  # 定义样本矩阵X的降维矩阵Z

        self.centerX = self._centralized()  # 生成矩阵X的中心化矩阵
        self.C = self._cov()  # 生成协方差矩阵
        self.U = self._U()  # 生成转化矩阵
        self.Z = self._Z()  # 生成最后的降维矩阵，Z = X * U

    # 样本矩阵X中心化
    def _centralized(self):
        print('样本矩阵X：\n', self.X)

        '''
        通过列表生成式，依次计算每个特征的均值。最后转换成Numpy数组方便后续计算
        生成的数组是一维 (n,) 类似于 (1,n)
        '''
        mean = np.array([np.mean(arr) for arr in self.X.T])
        print('样本集的特征均值：\n', mean)

        # 矩阵中心化
        centrX = self.X - mean  # 能直接相减是因为Numpy实现了广播机制，可以直接将较小维度数组扩展为与较大维度数组具有相同的形状
        print('样本矩阵X的中心化矩阵centrX：\n', centrX)
        return centrX

    # 求样本矩阵X的协方差矩阵
    def _cov(self):
        # 样本集的总样例数（行数）
        ns = self.centerX.shape[0]

        # 通过公式求样本矩阵X的协方差矩阵C
        C = np.dot(self.centerX.T, self.centerX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C：\n', C)
        return C

    # 求样本矩阵X降维的转换矩阵U
    def _U(self):
        '''
        样本矩阵X降维的转换矩阵U，shape(n,k), n是矩阵X的特征维度总数，K是降维矩阵的特征维度
        '''

        # 求协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)  # 特征值赋值给a（一维数组），对应特征向量赋值给b（二维数组），如列向量 b[:,i] 对应于 特征值a[i]
        print('样本集的协方差矩阵C的特征值：\n', a)
        print('样本集的协方差矩阵C的特征向量：\n', b)

        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-a)  # 默然是按照升序，降序排列添加负号

        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)  # 取前K个特征值（按从大到小排序）对应的特征向量形成的降维矩阵，每一列即一个特征向量
        print('降维转换矩阵U：\n', U)
        return U

    # 生成最终的降维矩阵Z
    def _Z(self):
        '''
        按照 Z = X*U 求降维矩阵U，n是样本总数，K是降维矩阵中特征维度总数
        '''
        Z = np.dot(self.X, self.U)
        print('X shape', self.X.shape)
        print('U shape', self.U.shape)
        print('Z shape', Z.shape)
        print('样本矩阵X的降维矩阵Z：\n',Z)

        return Z


if __name__ == '__main__':
    '''
    样本矩阵10x3,即样例为10，特征维度是3
    '''
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

    K = X.shape[1] - 1
    print('样本矩阵：\n', X)
    pca = CPCA(X, K)
