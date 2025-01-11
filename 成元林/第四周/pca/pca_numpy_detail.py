import numpy as np

def mean(X):
    """
    对原始数据均值化。去中心化，说明：每个样本为一行，每一列代表是一个特征，所以要每一行的没一列的数据，减去平均值
    @param X: 原始数据矩阵
    @return:
    """
    # 每一列求平均值
    mean = np.array([np.mean(item) for item in X.T])
    X = X-mean
    print("去中心化后矩阵\n",X)
    return X;

def cov_martrix(m,D):
    """
    求协方差矩阵，说明，根据公示Z = D的转置*D/(m-1)
    @param m: 样本数
    @param D: 去中心化后的数据
    @return:
    """
    cov = np.dot(D.T,D)/(m-1)
    print("协方差矩阵:\n",cov)
    return cov

def get_eigen_v(cov_x):
    """
    获取协方差特征值与特征向量
    @param cov_x: 协方差矩阵
    @return:
    """
    eigen_value,eigen_vector = np.linalg.eig(cov_x)
    print("特征值\n：",eigen_value)
    print("特征向量\n：",eigen_vector)
    return (eigen_value,eigen_vector)
if __name__ == '__main__':
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    # 去中心化
    X_av = mean(X)
    print(X_av.shape)
    K = X_av.shape[1]-1
    # 求协方差矩阵
    cov_x = cov_martrix(X_av.shape[0],X_av)
    # 获取特征值与特征向量
    (eigen_value,eigen_vector) = get_eigen_v(cov_x)
    # 特征值索引排序降序
    sort = np.argsort(-1*eigen_value)
    print(sort[:K])
    # 取得k维的特征向量
    reduce_eigen_vector = eigen_vector[:,sort[:K]]
    # 降维后的特征向量
    print("降维后的特征向量\n",reduce_eigen_vector)
    #数据投影到特征向量，这里需要注意是去中心化后的数据
    pca_data = np.dot(X_av,reduce_eigen_vector)
    print("样本x的降维矩阵\n",pca_data)
