
import numpy as np


class CPCA(object):

    def __init__(self, sample_set, target_size):
        '''
        :param sample_set: 样本矩阵
        :param target_size: 样本矩阵要特征降维成多少阶的
        '''
        self.sample_set = sample_set
        self.target_size = target_size
        self.center_matrix = []    # 样本矩阵的中心化处理
        self.covariance_matrix = []      # 样本矩阵的协方差矩阵
        self.eigenvector_reduction_matrix = []      # 样本矩阵的特征向量降维变换矩阵
        self.final_matrix = []      # 降维后的样本矩阵

        self.center_matrix = self._center_func()
        self.covariance_matrix = self._covariance_func()
        self.eigenvector_reduction_matrix = self._eigenvector_reduction_func()
        self.final_matrix = self._final_func()

    def _center_func(self):
        '''中心化'''
        _center_matrix = []
        # 计算样本集的特征均值
        _mean = np.array([np.mean(attr) for attr in self.sample_set.T])
        print('样本集的特征均值:\n', _mean)
        # 中心化处理
        _center_matrix = self.sample_set - _mean
        print('样本矩阵的中心化_center_matrix:\n', _center_matrix)
        return _center_matrix

    def _covariance_func(self):
        '''求协方差矩阵'''
        num = self.sample_set.shape[0]
        _covariance_matrix = np.dot(self.center_matrix.T, self.center_matrix) / (num - 1)
        print('样本矩阵的协方差矩阵_covariance_matrix:\n', _covariance_matrix)
        return _covariance_matrix

    def _eigenvector_reduction_func(self):
        '''特征向量降维转换矩阵'''
        # 先求协方差矩阵的特征值和特征向量
        _eigen_value, _eigen_vector = np.linalg.eig(self.covariance_matrix)
        print('样本集的协方差矩阵C的特征值:\n', _eigen_value)
        print('样本集的协方差矩阵C的特征向量:\n', _eigen_vector)
        # 给出特征值降序的top的索引序列
        index_arr = np.argsort(-1 * _eigen_value)
        # 构建target_size阶降维的降维转换矩阵
        eigenvector_reduction_t = []
        for i in range(self.target_size):
            eigenvector_reduction_t.append(_eigen_vector[:, index_arr[i]])
        _eigenvector_reduction_matrix = np.transpose(eigenvector_reduction_t)
        print('%d阶降维转换矩阵_eigenvector_reduction_matrix:\n' % self.target_size, _eigenvector_reduction_matrix)
        return _eigenvector_reduction_matrix

    def _final_func(self):
        '''降维后的样本矩阵'''
        _final_matrix = np.dot(self.sample_set, self.eigenvector_reduction_matrix)
        print('sample_set shape:', np.shape(self.sample_set))
        print('eigenvector_reduction_matrix shape:', np.shape(self.eigenvector_reduction_matrix))
        print('final_matrix shape:', np.shape(_final_matrix))
        print('样本矩阵X的降维矩阵_final_matrix:\n', _final_matrix)
        return _final_matrix


if __name__ == '__main__':
    # 样本集
    sample_set = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    # 样本要特征降维成 几阶的
    target_size = sample_set.shape[1] - 1
    print(f'样本集({sample_set.shape[0]}行{sample_set.shape[1]}列，{sample_set.shape[0]}个样例，每个样例{sample_set.shape[1]}个特征):\n{sample_set}')
    pca = CPCA(sample_set, target_size)