import numpy as np

class PCA():
    def __init__(self,n_components):
        self.n_components = n_components

    def fit_transform(self,X):
        self.n_features = X.shape[1]  # 存储X的特征数量
        # 中心化处理
        X = X - X.mean(axis =0)  # 求特征的均值，并从原始数据中减去
                                # axis = 0 是求列（特征）的均值        # axis = 1 是求行的均值
        # 求协方差矩阵
        self.covMatrix = np.dot(X.T,X)/X.shape[0]  # X的转置*X，再除以样本数量——>得到协方差矩阵

        # 计算协方差矩阵的特征值和特征向量
        cov_value,cov_vectors = np.linalg.eig(self.covMatrix)

        # 对特征值进行降序排列
        desc_cov_value = np.argsort(-cov_value)

        # 选择前n_components个最大的特征值对应的特征向量作为降维的转换矩阵（即主成分）
        self.transformMatrix = cov_vectors[:,desc_cov_value[:self.n_components]]

        # X和降维转换矩阵相乘得到降维后的矩阵  Z=XU
        self.Ddata = np.dot(X,self.transformMatrix)
        return self.Ddata

pca = PCA(n_components=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
PCA_X = pca.fit_transform(X)
print(PCA_X)