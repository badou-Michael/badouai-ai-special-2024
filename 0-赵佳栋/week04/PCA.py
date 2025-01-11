'''
@Project ：BadouCV 
@File    ：PCA_numpy_detail.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/17 15:29 
'''
import numpy as np

class PCA:
    """
    主成分分析类

    属性：
        components_: ndarray
            降维后的特征向量
        explained_variance_ratio_: ndarray
            每个主成分解释的方差比例
        mean_: ndarray
            原始数据每一维的均值

    方法：
        fit(X):
            对数据进行中心化并计算主成分
        transform(X):
            将数据投影到新的特征空间
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        """
        对数据进行中心化并计算主成分

        参数：
            X: ndarray, shape (n_samples, n_features)
                训练数据
        """

        # 中心化数据
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 对特征值和特征向量进行排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 选择前n_components个主成分
        if self.n_components is None:
            self.n_components = X.shape[1]
        self.components_ = eigenvectors[:, :self.n_components]

        # 计算每个主成分解释的方差比例
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance

        return self

    def transform(self, X):
        """
        将数据投影到新的特征空间

        参数：
            X: ndarray, shape (n_samples, n_features)
                待转换的数据

        返回：
            X_transformed: ndarray, shape (n_samples, n_components)
                转换后的数据
        """

        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
