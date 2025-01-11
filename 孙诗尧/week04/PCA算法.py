# 1.对每个特征的样本数据进行去中心化
# 2.计算协方差矩阵
# 3.计算协方差矩阵的特征值和特征向量，特征值最大的方向代表方差最大，将特征值从大到小排列，取前k个对应的特征向量排列形成一个矩阵
# 4.将原始数据投影到新坐标系上，得到降维的数据
import numpy as np
from sklearn.datasets import load_iris


class PCA:
    def __init__(self, k_input):
        self.k = k_input

    # 对每个特征的样本数据进行去中心化
    def centralize(self, matrix_input):
        # 求每一列（即每个特征的样本）的平均值
        characteristic_mean = np.mean(matrix_input, axis=0)
        # 利用NumPy的广播机制直接进行矩阵减法，得到去中心化后的样本矩阵
        self.matrix_centralized = matrix_input - characteristic_mean
        return self.matrix_centralized

    # 计算协方差矩阵
    def covmatrix(self):
        # 计算得到样本矩阵的协方差矩阵，四个特征得到4*4的协方差矩阵
        self.matrix_cov = np.cov(self.matrix_centralized.T)
        return self.matrix_cov

    # 计算降维矩阵
    def eigmatirx(self):
        character, vector = np.linalg.eig(self.matrix_cov)
        # 将特征值从大到小排序
        character_index = np.argsort(-character)
        # 取前k个最大的特征值所对应的特征向量按序排序形成降维矩阵
        self.dimension_reduction = vector[:, character_index[:self.k]]
        return self.dimension_reduction

    # 求得降维后的数据
    def dimreduction(self):
        return np.dot(self.matrix_centralized, self.dimension_reduction)


if __name__ == "__main__":
    k = 2
    # 加载鸢尾花数据集
    iris = load_iris()
    # 得到鸢尾花数据集的样本矩阵
    matrix = iris.data
    X = PCA(2)
    X.centralize(matrix)
    X.covmatrix()
    X.eigmatirx()
    print(X.dimreduction())

"""
k = 2
# 加载鸢尾花数据集
iris = load_iris()
# 得到鸢尾花数据集的样本矩阵
matrix = iris.data
# 求每一列（即每个特征的样本）的平均值
characteristic_mean = np.mean(matrix, axis=0)
# 利用NumPy的广播机制直接进行矩阵减法，得到去中心化后的样本矩阵
matrix_centralized = matrix - characteristic_mean
# 计算得到样本矩阵的协方差矩阵，四个特征得到4*4的协方差矩阵
matrix_cov = np.cov(matrix_centralized.T)
# 计算协方差矩阵的特征值和特征向量
character, vector = np.linalg.eig(matrix_cov)
# 将特征值从大到小排序
character_index = np.argsort(-character)
# 取前k个最大的特征值所对应的特征向量按序排序形成降维矩阵
dimension_reduction = vector[:, character_index[:k]]
# 得到降维后的矩阵
matrix_reduction = np.dot(matrix_centralized, dimension_reduction)
print(dimension_reduction)
"""
