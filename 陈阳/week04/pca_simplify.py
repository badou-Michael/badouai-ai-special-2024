import numpy as np


class PCA(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features = X.shape[1]
        X = X - np.mean(X, axis=0)
        self.covariance = np.dot(X.T, X) / X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components = eig_vectors[:, idx[:self.n_components]]
        return np.dot(X, self.components)


# 调用
pca = PCA(n_components=2)
X = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
newX = pca.fit_transform(X)
print(newX)  # 输出降维后的数据
