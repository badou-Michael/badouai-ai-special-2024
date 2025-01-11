# -*- coding: utf-8 -*-
import numpy as np

class CPCA(object):
    def __init__(self, src, k):
        '''
        src: 样本矩阵
        k: 降维目标阶数
        '''
        self.src = src
        self.k = k
        self.centrX = self.__centralized()
        self.C = self.__getCov()   # 协方差矩阵
        self.U = self.__getTrans()     # 降维转换矩阵
        self.dst = self.__getRst()     # dst = src*U 求得

    def __centralized(self):
        '''矩阵src的中心化'''
        # 获取特征均值
        mean = np.array([np.mean(attr) for attr in self.src.T])
        print('样本集的特征均值:\n',mean)
        centrSrc = self.src - mean
        print('样本矩阵的中心化:\n', centrSrc)
        return centrSrc

    def __getCov(self):
        '''求协方差矩阵'''
        #样本总数
        num = np.shape(self.src)[0]
        cov = np.dot(self.centrX.T, self.centrX)/(num - 1)
        print('样本矩阵的协方差矩阵:\n', cov)
        return cov

    def __getTrans(self):
        '''降维转换矩阵'''
        a, b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 获取降序索引
        idx = np.argsort(-1*a)
        # 按降序按列提取前k个, 会将列转化成行向量
        trans = [b[:, idx[i]] for i in range(self.k)]
        # 因此需要再转置一下
        transT = np.transpose(trans)
        return transT

    def __getRst(self):
        '''dst = src*U 求降维矩阵'''
        dst = np.dot(self.src, self.U)
        print('样本矩阵的降维矩阵:\n', dst)
        return dst

if __name__ == '__main__':
    # 10样本3特征的样本集, 行为样例，列为特征维度
    src = np.array([
        [10, 15, 29],
        [15, 46, 13],
        [23, 21, 30],
        [11, 9,  35],
        [42, 45, 11],
        [9,  48, 5],
        [11, 21, 14],
        [8,  5,  15],
        [11, 12, 21],
        [21, 20, 25]
    ])
    k = np.shape(src)[1] - 1  # 3维降为2维
    pca = CPCA(src, k)