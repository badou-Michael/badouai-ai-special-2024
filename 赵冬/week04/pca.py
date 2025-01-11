import numpy as np


class PCA(object):
    def __init__(self, n_components):
        self.n_components = n_components;

    def fit_transform(self, X):
        # 去中心化
        center_x = X - X.mean(axis=0)
        # 协方差
        cov = np.dot(center_x.T, center_x)/X.shape[0]
        # 特征值和特征向量
        vals, vectors = np.linalg.eig(cov)
        nid = np.argsort(-1*vals)
        # 降维矩阵
        components = vectors[:, nid[:self.n_components]]
        # X数据降维
        return np.dot(X, components)

if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
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
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(K)
    print(pca.fit_transform(X))
