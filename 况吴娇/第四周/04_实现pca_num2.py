#coding=utf-8

import numpy as np

class PCA():
    def __init__(self,n_components):
        self.n_components = n_components
    def fit_transform(self,X):
        self.n_features=X.shape[1]#存储了数据集的特征数量。
        #协方差
        X=X-X.mean(axis=0)#X 的每个元素中减去对应 列 的均值，得到中心化后的数据。
        self.convariance=np.dot(X.T,X)/X.shape[0]
        # 不使用 self 前缀，那么每次调用方法时，都会创建一个新的局部变量，之前存储的数据会丢失。如果你希望在类的多个方法之间共享数据，不使用 self 前缀将无法实现
        # 求协方差矩阵的特征值和特征向量
        a, b = np.linalg.eig(self.convariance)
        # 获得降序排列特征值的序号
        ind=np.argsort(-1*a)
        # 降维矩阵
        self.components_t =b[:,ind[:self.n_components]]
        # 对X进行降维
        return  np.dot(X,self.components_t)

#调用
pca=PCA(n_components=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
newX=pca.fit_transform(X)
print(newX)                  #输出降维后的数据


