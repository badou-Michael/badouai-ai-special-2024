#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/6 20:57
@Author  : Mr.Long
"""
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg as sl


class HomeworkW7(object):

    def __init__(self, nd_array, t):
        self.nd_array = nd_array
        self.t = t

    def hierarchical_cluster(self):
        """
        层次聚类
        """
        z = linkage(self.nd_array, method='ward')
        fcluster(z, self.t, criterion='distance')
        dendrogram(z)
        plt.show()


def rewrite_ransac(data, model, n, k, t, d, debug=False, return_all=False):
    iterations = 0
    bestfit = None
    best_err = np.inf  # 设置默认值
    best_lie_idx = None
    while iterations < k:
        maybe_idx, test_idx = random_partition(n, data.shape[0])
        print('test_idx = ', test_idx)
        maybe_lie = data[maybe_idx, :]  # 获取size(maybe_idx)行数据(Xi,Yi)
        test_points = data[test_idx]  # 若干行(Xi,Yi)数据点
        maybe_model = model.fit(maybe_lie)  # 拟合模型
        test_err = model.get_error(test_points, maybe_model)  # 计算误差:平方和最小
        print('test_err = ', test_err < t)
        also_idx = test_idx[test_err < t]
        print('also_idx = ', also_idx)
        also_lie = data[also_idx, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(also_lie) = %d' % (iterations, len(also_lie)))
        print('d = ', d)
        if len(also_lie) > d:
            better_data = np.concatenate((maybe_lie, also_lie))  # 样本连接
            better_model = model.fit(better_data)
            better_errs = model.get_error(better_data, better_model)
            this_err = np.mean(better_errs)  # 平均误差作为新的误差
            if this_err < best_err:
                bestfit = better_model
                best_err = this_err
                best_lie_idx = np.concatenate((maybe_idx, also_idx))  # 更新局内点,将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return bestfit, {'lie': best_lie_idx}
    else:
        return bestfit

def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idx = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_idx)  # 打乱下标索引
    idx_s1 = all_idx[:n]
    idx_s2 = all_idx[n:]
    return idx_s1, idx_s2


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        data_a = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        data_b = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, res_ids, rank, s = sl.lstsq(data_a, data_b)  # residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        a_x = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        b_x = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        b_fit = sp.dot(a_x, model)  # 计算的y值,b_fit = model.k*a_x + model.b
        err_per_point = np.sum((b_x - b_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    a_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    b_exact = sp.dot(a_exact, perfect_fit)  # y = x * k
    print(b_exact)
    # 加入高斯噪声,最小二乘能很好的处理
    a_noisy = a_exact + np.random.normal(size=a_exact.shape)  # 500 * 1行向量,代表Xi
    b_noisy = b_exact + np.random.normal(size=b_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idx = np.arange(a_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idx)  # 将all_idx打乱
        outlier_idx = all_idx[:n_outliers]  # 100个0-500的随机局外点
        a_noisy[outlier_idx] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        b_noisy[outlier_idx] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((a_noisy, b_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, res_id, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, ransac_data = rewrite_ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    if 1:
        import pylab

        sort_idx = np.argsort(a_exact[:, 0])
        a_col0_sorted = a_exact[sort_idx]  # 秩为2的数组
        if 1:
            pylab.plot(a_noisy[:, 0], b_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(a_noisy[ransac_data['lie'], 0], b_noisy[ransac_data['lie'], 0], 'bx',
                       label="RANSAC data")
        # else:
        #     pylab.plot(a_noisy[non_outlier_idxs, 0], b_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
        #     pylab.plot(a_noisy[outlier_idx, 0], b_noisy[outlier_idx, 0], 'r.', label='outlier data')

        pylab.plot(a_col0_sorted[:, 0],
                   np.dot(a_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(a_col0_sorted[:, 0],
                   np.dot(a_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(a_col0_sorted[:, 0],
                   np.dot(a_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    test()
    # nd_array = np.array([[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]])
    # homework_w7 = HomeworkW7(nd_array, 4)
    # homework_w7.hierarchical_cluster()


