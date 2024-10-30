
import numpy as np
'''
手写pca实现
1.中心化：将原图的每个特征（维度）减去其均值
2.计算协方差矩阵
3.计算特征值和特征向量
4.按特征值从大到小排序，并选择前X个特征向量
5.将数据做映射

'''

class PCA:
    def __init__(self, n_components):
        """
        :param n_components: 降维，保留剩余特征

        """
        self.n_components=n_components
        self.mean=None
        self.components=None

    def fit(self,X):
        # Step 1:标准化数据
        self.mean=np.mean(X,axis=0) #np.mean 返回数组元素的平均值
        X_centered=X-self.mean

        # Step 2:计算协方差矩阵
        """
         np.cov 跟 np.dot的区别在于
         np.dot(X.T,X)/X.shape[0] 需要先手动中心化数据，然后计算中心化数据的点积，最后除以样本数(行)
        """
        cov_matrix = np.cov(X_centered, rowvar=False)
        # Step 3: 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Step 4: 按特征值从大到小排序，选择前n_components个特征向量
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]

    def transform(self, X):
        # Step 5: 将数据投影到主成分上
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)


if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    pca = PCA(n_components=2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("原始数据的形状：", X.shape)
    print("降维后数据的形状：", X_projected.shape)
    print('样本矩阵X的降维矩阵Z:\n', X_projected)
