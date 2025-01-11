import numpy as np

def Calculate_mean(matrix):
    mean = np.mean(matrix, axis=0)
    print("矩阵的均值为：\n", mean)
    # for i in range(np.shape(matrix)[]):
    result = matrix - mean
    print("矩阵中心化的值为：\n", result)
    return result

def Calculate_cov(matrix):
    num = np.shape(matrix)[0]
    cov = 1/num * np.dot(matrix.T, matrix)
    print("协方差矩阵：\n", cov)
    return cov

def Calculate_eig(matrix):
    # 求解特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print("特征值：\n", eigenvalues)
    print("特征向量：\n", eigenvectors)
    return eigenvalues, eigenvectors

def Calculate_dim(matrix, eigenvectors, k):
    k_result = np.dot(matrix, eigenvectors[:, :k])
    print("降维后的向量为：\n", k_result)
    return k_result

if __name__=='__main__':
    matrix = np.array([[10, 15, 29],
                        [15, 46, 13],
                        [23, 21, 30],
                        [11, 9,  35],
                        [42, 45, 11],
                        [9,  48, 5],
                        [11, 21, 14],
                        [8,  5,  15],
                        [11, 12, 21]])
    mean_result = Calculate_mean(matrix)
    cov_result = Calculate_cov(mean_result)
    _, eig_vector = Calculate_eig(cov_result)
    result = Calculate_dim(matrix, eig_vector, 2)
