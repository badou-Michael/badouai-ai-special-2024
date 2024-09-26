"""
PCA -yzc
"""

import numpy as np
class PCA():
    def __init__(self,n_components):
        self.n_components = n_components
    
    def fit_transform(self,X):
        self.n_features_ = X.shape[1]
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T,X)/X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals,eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:,idx[:self.n_components]]
        # 对X进行降维
        return np.dot(X,self.components_)
 
# 调用
pca = PCA(n_components=2)
X = np.array([[-10,2,23,-3], [23,12,8,-11], [-13,-8,15,-21], [12,19,32,11], [22,18,62,11], [23,35,43,52]])  #导入数据，维度为4
newX=pca.fit_transform(X)
print(newX)                  #输出降维后的数据
