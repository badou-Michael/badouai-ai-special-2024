# 1. 基于特征值和特征向量的PCA降维

import numpy as np
import matplotlib.pyplot as plt

def PCA(X):
    # 1. 数据标准化
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    new_X = (X - mean_X) / std_X

    # 2. 计算协方差矩阵
    cov_X = np.cov(new_X, rowvar=False) # rowvar=False表示每一行代表一个变量

    # 3. 计算特征值和特征向量
    eig_vals, eig_vecs = np.linalg.eigh(cov_X)

    # 4. 按照特征值大小排序
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # 5. 选择前2个主成分
    n_components = 2
    eigvec_subset = eig_vecs[:, :n_components]
    # 6. 将原始数据转换到新空间
    X_pca = np.dot(new_X, eigvec_subset)
    '''
    np.dot 是 NumPy 库提供的函数，用于计算两个数组的点积（矩阵乘法）
    当 new_X（大小为 100x5，100个样本，5个特征）与 eigvec_subset（大小为 5x2，5个特征向量，2个主成分）进行点积时，
    结果 X_pca 将是大小为 100x2 的矩阵，表示每个样本在前两个主成分上的投影
    '''
    return X_pca

# 测试PCA降维
np.random.seed(42)
X = np.random.rand(100, 5)
X_pca= PCA(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.arange(100), cmap='rainbow')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA by eig')
plt.colorbar(label='Class')
# plt.savefig('PCA_eig.png')
plt.show()



# 2. sklearn实现PCA降维
from sklearn.decomposition import PCA

np.random.seed(42)
X = np.random.rand(100, 5)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.arange(100), cmap='rainbow')
''' 
X_pca是经过PCA处理后的数据，其中的每一行代表一个样本，每一列代表一个主成分
X_pca[:, 0]即取出所有样本在第一个主成分上的值
X_pca[:, 1]选择所有样本在第二个主成分上的值
'''
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA by sklearn')
plt.colorbar(label='Class')
# plt.savefig('PCA_sklearn.png')
plt.show()


# 鸢尾花数据集PCA降维
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='rainbow')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA on Iris dataset')
plt.colorbar(label='Class')
plt.savefig('PCA_iris.png')
plt.show()

