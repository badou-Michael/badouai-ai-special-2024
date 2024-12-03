# 实现PCA
# author:苏百宣

import numpy as np

class PCA():
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


# 调用
pca = PCA(n_components=3)
X = np.array(
    [[81, -97, -78, -96, -5, 96],
     [64, 39, 87, -89, -92, -9],
     [57, -3, -7, -57, 34, -61],
     [-66, 73, -34, -92, 86, 0],
     [-26, 9, 99, 40, -73, 31],
     [-74, 93, -85, 9, -30, -84],
     [88, -73, -95, -31, 72, -12],
     [94, 44, 94, -61, -79, 84],
     [-45, -65, 11, 42, 89, 33],
     [77, -98, -19, -51, 28, -76]])
newX = pca.fit_transform(X)
print(newX)
