#coding=utf-8
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4

# 直接调用封装好的函数实现
pca = PCA(n_components=2)   #降到2维
pca.fit(X)                  #执行
newX=pca.fit_transform(X)   #降维后的数据
print(newX)                  #输出降维后的数据


# PCA实现方法
def fit_transform(X, n_components):
    X = X - X.mean(axis=0)
    covariance = np.dot(X.T, X) / X.shape[0]
    eig_vals, eig_vectors = np.linalg.eig(covariance)
    idx = np.argsort(-eig_vals)
    components_ = eig_vectors[:, idx[:n_components]]
    return np.dot(X, components_)

newX2 = fit_transform(X, n_components=2)
print(newX2)
