import numpy as np

# 自定义PCA算法

class PCA():
    def __init__(self,n_components):
        self.n_components=n_components

    def fit_transform(self,X):
        self.features=X.shape[1]
        X=X-X.mean(axis=0) #去中心化
        self.covariance=np.dot(X.T,X)/X.shape[0] #求协方差矩阵
        eigen_values,eigen_vectors=np.linalg.eig(self.covariance)# 求协方差矩阵的特征值和特征向量
        indexs=np.argsort(-eigen_values) #取反 获得降序排序的特征值的下标的数组
        self.eigen_vectors_compos=eigen_vectors[:,indexs[:self.n_components]] #取出对应的排在前n_components个特征值对应的特征向量
        return np.dot(X,self.eigen_vectors_compos)


X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]]) #导入数据，维度为4
pca=PCA(n_components=2)
X_2=pca.fit_transform(X)
print(X_2)

