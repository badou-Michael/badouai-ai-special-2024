import numpy as np


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # 特征有几维
        self.n_features = X.shape[1]

        # 中心化
        X = X - X.mean(axis=0)
        # 求协方差矩阵
        self.covariance = np.dot(X.T, X) / X.shape[0]  # X.shape[0]:样本总数

        # 求协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance)

        # 获得降序排列特征值的序号
        idx = np.argsort(-eigenvalues)

        # 降维矩阵
        self.components_ = eigenvectors[:, idx[:self.n_components]]

        # 对X进行降维
        return np.dot(X, self.components_)


# 调用
pca = PCA(n_components=2)
# 导入数据，维度为4
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
newX = pca.fit_transform(X)
# 输出降维后的数据
print(newX)
