import numpy as np

class PCA(object):

    def __init__(self, sample, dimension):
        self.sample_matrix = sample
        self.after_reduction_dim = dimension
        self.central_matrix = []
        self.covariance_matrix = []
        self.reduction_dim_transformer = []

    def decentration(self):
        feature_means = np.array([np.mean(single_feature_values) for single_feature_values in self.sample_matrix.T])
        self.central_matrix = self.sample_matrix - feature_means
        return self.central_matrix


    def covariance_compute(self):
        self.covariance_matrix = np.dot(self.central_matrix.T, self.central_matrix) / (len(self.sample_matrix) - 1)
        return self.covariance_matrix

    # 每一列对应于矩阵, A 的一个特征向量，列的顺序与特征值的顺序对应
    def egi_compute(self):
        egi_val, egi_vec = np.linalg.eig(self.covariance_compute())
        answer = np.argsort(-1 * egi_val)
        reduction_matrix_T = [egi_vec[:, answer[i]] for i in range(self.after_reduction_dim)]
        self.reduction_dim_transformer = np.transpose(reduction_matrix_T)
        return self.reduction_dim_transformer

    def result(self):
        return np.dot(self.sample_matrix, self.reduction_dim_transformer)

if __name__ == '__main__':
    # 总共有10个样本，每个样本有4个特征。
    X = np.array([[10, 15, 29, 21],
                  [15, 46, 13, 34],
                  [23, 21, 30, 12],
                  [11, 9,  35, 24],
                  [42, 45, 11, 9],
                  [9,  48, 5, 54],
                  [11, 21, 14, 37],
                  [8,  5,  15, 32],
                  [11, 12, 21, 29],
                  [21, 20, 25, 56]])

    pca = PCA(X, 3)
    centrial_matrix = pca.decentration()
    covariance_matrix = pca.covariance_compute()
    reduction_dim_transformer = pca.egi_compute()
    result = pca.result()
    print(result)
