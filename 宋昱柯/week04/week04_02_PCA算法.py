import numpy as np

class PCA:
    """
    PCA算法

    """

    def __init__(self, k):
        """X.shape=(num_samples,num_attributes)"""
        self.k = k

    def _centralize(self):
        """样本数据中心化"""
        mean = np.array([np.mean(attr) for attr in self.X.T])
        # print(mean)
        # print(self.X - mean)
        return self.X - mean

    def _cov(self):
        """求协方差矩阵"""
        return np.dot(self.X_centralized.T, self.X_centralized) / (
            self.X_centralized.shape[0] - 1
        )

    def __call__(self, X):
        self.X = X
        # 中心化
        self.X_centralized = self._centralize()
        # 选取特征向量
        eigenvalues, eigenvectors = np.linalg.eig(self._cov())
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        selected_components = sorted_eigenvectors[:, : self.k]
        # 计算投影
        X_pca = np.dot(self.X, selected_components)

        print("选择的特征向量为：\n", selected_components)
        print("投影后的矩阵为：\n", X_pca)

        return X_pca

if __name__ == "__main__":
    "10样本3特征的样本集, 行为样例，列为特征维度"
    X = np.array(
        [
            [10, 15, 29],
            [15, 46, 13],
            [23, 21, 30],
            [11, 9, 35],
            [42, 45, 11],
            [9, 48, 5],
            [11, 21, 14],
            [8, 5, 15],
            [11, 12, 21],
            [21, 20, 25],
        ]
    )
    K = np.shape(X)[1] - 1
    print("样本集(10行3列，10个样例，每个样例3个特征):\n", X)
    pca = PCA(K)
    X_pca = pca(X)
