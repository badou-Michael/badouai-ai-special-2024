import numpy as np
from sklearn.decomposition import PCA

def covariance_matrix(X):
    """计算数据集X的协方差矩阵"""
    mean_vector = np.mean(X, axis=0)
    demeaned_X = X - mean_vector
    return demeaned_X, np.dot(demeaned_X.T, demeaned_X) / (X.shape[0] - 1)


def eigenvectors(covariance_matrix):
    """计算协方差矩阵的特征向量"""
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    return eigenvalues, eigenvectors


def PCANew(X, n_components):
    """执行PCA，保留k个主成分"""
    demeaned_X, covariance = covariance_matrix(X)
    eigenvalues,vectors = eigenvectors(covariance)

    sort_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvalues = eigenvalues[sort_indices[:n_components]]
    top_eigenvectors = vectors[:, sort_indices[:n_components]]
    # 5. 投影数据
    X_transformed = np.dot(demeaned_X, top_eigenvectors)

    return X_transformed * -1, top_eigenvalues, top_eigenvectors


# 示例用法
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 20, 33]])
trans, values, vectors = PCANew(data, 2)
print("数据投影结果:\n", trans)
# print("选择的特征向量:\n", eigenvector_res)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(data)
print('reduced', X_reduced)

