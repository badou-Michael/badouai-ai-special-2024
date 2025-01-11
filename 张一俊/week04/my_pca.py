import numpy as np
from sklearn.decomposition import PCA
# from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 法一：纯手写实现
def pca_manual_1(X, K):
    # 1. 样本矩阵中心化
    mean = np.array([np.mean(attr) for attr in X.T])
    centrX = X - mean
    # print(centrX)

    # 2. 样本矩阵的协方差矩阵
    ns = np.shape(centrX)[0]
    C = np.dot(centrX.T, centrX) / (ns - 1)

    # 3. 协方差矩阵的特征值和特征向量
    a, b = np.linalg.eig(C)
    # print(a, b)

    # 4. 降序排列特征值和特征向量
    ind = np.argsort(-1*a)
    # print(ind)
    UT = [b[:, ind[i]] for i in range(K)]
    U = np.transpose(UT)
    # print(U)

    # 5. 降维矩阵Z
    Z = np.dot(centrX, U)

    return Z

# 法二：使用np的mean、dot等实现
def pca_manual_2(X, n_components):
    # 1. 样本矩阵中心化
    centrX = X - X.mean(axis=0)  # X - np.mean(X, axis=0)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # 2. 样本矩阵的协方差矩阵
    covariance = np.dot(centrX.T, centrX) / centrX.shape[0]
    
    # 3. 协方差矩阵的特征值和特征向量
    eig_vals, eig_vectors = np.linalg.eig(covariance)

    # 4. 获得降维矩阵
    idx = np.argsort(-eig_vals)

    components_ = eig_vectors[:, idx[:n_components]]

    # 5. 对X进行降维
    return np.dot(centrX, components_)


def pca_manual_3(X, n_components):
    # 1. 数据标准化
    X_meaned = X - np.mean(X, axis=0)

    # 2. 计算协方差矩阵
    covariance_matrix = np.cov(X_meaned, rowvar=False)

    # 3. 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 4. 排序特征值和特征向量
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 5. 选择前n_components个特征向量
    eigenvector_subset = sorted_eigenvectors[:, :n_components]

    # 6. 转换数据
    X_pca = X_meaned.dot(eigenvector_subset)
    return X_pca

# 法三：调用scikit-learn的PCA接口实现
def pca_interface_1(X, n_components):
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    return X_pca


if __name__ == "__main__":
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]]) # load_iris数据集

    # data = load_iris()
    # X = data.data

    X_pca = pca_manual_1(X, K=2)
    print(X_pca), print()  # 四个结果符号会相反是因为求解特征值特征向量时用了不同符号而已

    X_pca = pca_manual_2(X, n_components=2)
    print(X_pca), print()

    X_pca = pca_manual_3(X, n_components=2)
    print(X_pca), print()

    X_pca = pca_interface_1(X, n_components=2)
    print(X_pca), print()


