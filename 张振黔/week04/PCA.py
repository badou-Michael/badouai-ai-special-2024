import numpy as np
def pca(X, n_components):

    # Step 1: 数据标准化
    mean = np.mean(X, axis=0)
    X_demean = X - mean
    std = np.std(X_demean, axis=0)
    X_std = X_demean / std

    # Step 2: 计算协方差矩阵
    covariance_matrix = np.cov(X_std.T)

    # Step 3: 求解协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4: 选择包含主要信息的主成分
    sorted_index = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_index]
    X_pca = np.dot(X_std, eigenvectors[:, 0:n_components])

    return X_pca

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
n_components = 2  # 降维后的维度
X_pca = pca(X, n_components)
print("降维后的数据集：")
print(X_pca)
