import numpy as np
from sklearn.decomposition import PCA


# 使用PCA求样本矩阵X的K阶降维矩阵Z

class CPCA(object):
    # X,样本矩阵X,K,X的降维矩阵的阶数，即X要特征降维成k阶

    def __init__(self, X, K):
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centerX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centerX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    # 矩阵X的中心化
    def _centralized(self):
        print("样本矩阵X:\n", self.X)
        centerX = []
        # 遍历self.X数组转置后的每一列，计算每一列的平均值
        # 样本矩阵中每一列是一个特征，所以要计算每一列的均值来中心化
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('样本集的特征均值:\n', mean)
        centerX = self.X - mean
        print('样本矩阵X的中心化centrX:\n', centerX)

        return centerX

    # 求样本矩阵X的协方差矩阵C
    def _cov(self):
        # 样本集的样例总数
        ns = np.shape(self.centerX)[0]
        print('样本集的总数:\n', ns)

        # 样本矩阵的协方差矩阵C
        # 计算中心化数据的转置与其自身的点积，得到一个形状为(m, m)的矩阵。
        # 将这个矩阵除以ns - 1，得到协方差矩阵
        # np.dot: 用于计算两个数组的点积。
        C = np.dot(self.centerX.T, self.centerX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    # 求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度
    def _U(self):
        a, b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)

        # 给出特征值降序的topK的索引降序序列 argsort默认升序
        ind = np.argsort(-1 * a)
        print("topK的索引序列:\n", ind)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        print("UT:\n", UT)
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    # 按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'
    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


class PCA2():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        # 对X进行降维
        return np.dot(X, self.components_)


if __name__ == "__main__":
    X = np.array([
        [12, 54, 21],
        [35, 51, 29],
        [42, 27, 53],
        [65, 23, 21],
        [32, 53, 45],
        [35, 45, 57],
        [45, 75, 65],
        [65, 54, 23],
        [25, 85, 78],
        [45, 35, 9],
    ])
    K = np.shape(X)[1] - 1
    print('k', K)
    print('np.shape(X)', np.shape(X))
    # print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)

    # 方法二的调用
    pca = PCA2(n_components=2)
    newX = pca.fit_transform(X)
    print('newX\n', newX)  # 输出降维后的数据

    # 方法三的调用
    pca = PCA(n_components=2)  # 降到2维
    pca.fit(X)  # 执行
    newX3 = pca.fit_transform(X)  # 降维后的数据
    print('newX3\n', newX3)  # 输出降维后的数据
