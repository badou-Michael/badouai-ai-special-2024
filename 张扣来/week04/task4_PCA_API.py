from operator import index

import numpy as np
class PCA():
    def __init__(self,n_components):
        self.n_components = n_components
    def fit_transform(self,X):
        # 元数据几列，几个特征
        self.n_features = X.shape[1]
        # 求协方差矩阵,延0轴，0代表行，取均值，参照mean函数
        X=X-X.mean(axis=0)
        # 协方差矩阵乘积
        self.covariance = np.dot(X.T,X)/X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        ind =np.argsort(-eig_vals)
        # 矩阵降维
        self.components_ = eig_vectors[:,ind[:self.n_components]]
        return np.dot(X,self.components_)
pca = PCA(n_components=3)#赋值降到几维
X = np.array([
    [12,24,56,7],
    [23,24,89,23],
    [13,9,78,1],
    [35,4,26,18],
    [18,8,7,24],
    [98,14,26,18],
    [48,24,37,29],
    [29,18,29,2]
    ])
newX = pca.fit_transform(X)
print(newX)

