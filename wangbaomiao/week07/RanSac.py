# -*- coding: utf-8 -*-
# time: 2024/11/5 16:12
# file: RanSac.py
# author: flame
import numpy as np
import scipy.linalg as sl
import scipy as sp
import matplotlib.pyplot as plt

"""
这是一个使用RANSAC算法进行线性拟合的示例代码。代码包括以下几个部分：
1. `random_poission` 函数：随机选择inliers和outliers的索引。
2. `ransac` 函数：实现RANSAC算法的核心逻辑。
3. `LinearLeastSquareModel` 类：定义线性最小二乘模型。
4. `test` 函数：生成测试数据并调用RANSAC算法进行拟合。
5. 主程序入口：调用 `test` 函数运行示例。
"""

def random_poission(n, n_data):
    """
    随机选择n个索引作为inliers，剩余的作为outliers。

    参数:
    n - int, 选择的inliers数量
    n_data - int, 数据集的大小

    返回:
    idxs1 - inliers的索引数组
    idxs2 - outliers的索引数组
    """
    """ 创建一个从0到n_data-1的索引数组。 """
    all_idxs = np.arange(n_data)
    """ 随机打乱索引数组。 """
    np.random.shuffle(all_idxs)
    """ 选择前n个索引作为inliers。 """
    idxs1 = all_idxs[:n]
    """ 剩余的索引作为outliers。 """
    idxs2 = all_idxs[n:]
    """ 返回inliers和outliers的索引数组。 """
    return idxs1, idxs2

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    实现RANSAC算法，用于鲁棒地拟合模型。
    参数:
    data - 输入数据，形状为 (N, D)，其中N是样本数，D是特征数
    model - 模型对象，需要实现fit和get_error方法
    n - 每次迭代选择的inliers数量
    k - 最大迭代次数
    t - 内点的最大误差阈值
    d - 内点的最小数量
    debug - 是否开启调试模式，默认为False
    return_all - 是否返回所有结果，默认为False
    返回:
    bestfit - 最佳拟合模型
    如果return_all为True，返回一个字典，包含最佳拟合模型和内点索引
    """
    """ 初始化迭代次数为0。 """
    iterations = 0
    """ 初始化最佳拟合模型为None。 """
    bestfit = None
    """ 初始化最佳误差为一个非常大的值。 """
    besterr = np.inf
    """ 初始化最佳内点索引为None。 """
    best_inlier_idxs = None
    """ 开始RANSAC迭代。 """
    while iterations < k:
        """ 随机选择n个inliers和剩余的outliers。 """
        maybe_idx, test_idx = random_poission(n, data.shape[0])
        """ 获取选定的inliers数据。 """
        maybe_inliers = data[maybe_idx]
        """ 获取剩余的outliers数据。 """
        test_points = data[test_idx]
        """ 使用选定的inliers数据拟合模型。 """
        maybe_model = model.fit(maybe_inliers)
        """ 计算outliers数据在拟合模型上的误差。 """
        test_err = model.get_error(test_points, maybe_model)
        """ 打印测试误差小于阈值t的布尔数组。 """
        print("test_err = ", test_err < t)
        """ 获取测试误差小于阈值t的outliers索引。 """
        also_idx = test_idx[test_err < t]
        """ 打印这些outliers的索引。 """
        print("also_idx = ", also_idx)
        """ 如果开启调试模式，打印一些调试信息。 """
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_idx)))
        """ 打印内点的最小数量d。 """
        print("d = ", d)
        """ 如果找到的内点数量大于d，进行进一步处理。 """
        if len(also_idx) > d:
            """ 将选定的inliers和新找到的内点合并。 """
            betterdata = np.concatenate((maybe_inliers, data[also_idx]))
            """ 使用合并后的数据重新拟合模型。 """
            bettermodel = model.fit(betterdata)
            """ 计算合并后数据在新拟合模型上的误差。 """
            better_errs = model.get_error(betterdata, bettermodel)
            """ 计算平均误差。 """
            thiserr = np.mean(better_errs)
            """ 如果当前模型的误差小于最佳误差，更新最佳模型和误差。 """
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idx, also_idx))
        """ 迭代次数加1。 """
        iterations += 1
    """ 如果没有找到合适的模型，抛出异常。 """
    if bestfit is None:
        raise ValueError("didn't meet fit acceptance criteria")
    """ 根据return_all参数决定返回的内容。 """
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit

class LinearLeastSquareModel:
    """
    定义线性最小二乘模型。
    参数:
    input_columns - 输入特征的列索引列表
    output_columns - 输出特征的列索引列表
    debug - 是否开启调试模式，默认为False
    """
    def __init__(self, input_columns, output_columns, debug=False):
        """ 初始化输入和输出特征的列索引。 """
        self.input_columns = input_columns
        self.output_columns = output_columns
        """ 初始化调试模式标志。 """
        self.debug = debug

    def fit(self, data):
        """
        使用输入数据拟合模型。
        参数:
        data - 输入数据，形状为 (N, D)，其中N是样本数，D是特征数
        返回:
        拟合得到的模型参数
        """
        """ 提取输入特征数据。 """
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        """ 提取输出特征数据。 """
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        """ 使用最小二乘法拟合模型。 """
        x, resids, rank, s = sl.lstsq(A, B)
        """ 返回拟合得到的模型参数。 """
        return x

    def get_error(self, data, model):
        """
        计算数据在给定模型上的误差。
        参数:
        data - 输入数据，形状为 (N, D)，其中N是样本数，D是特征数
        model - 模型参数
        返回:
        每个样本的误差
        """
        """ 提取输入特征数据。 """
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        """ 提取输出特征数据。 """
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        """ 计算模型预测值。 """
        B_fit = sp.dot(A, model)
        """ 计算每个样本的误差。 """
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        """ 返回每个样本的误差。 """
        return err_per_point

def test():
    """
    生成测试数据并调用RANSAC算法进行拟合。
    """
    """ 设置样本数量。 """
    n_samples = 500
    """ 设置输入特征数量。 """
    n_inputs = 1
    """ 设置输出特征数量。 """
    n_outputs = 1
    """ 生成精确的输入数据。 """
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    """ 生成精确的模型参数。 """
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    """ 生成精确的输出数据。 """
    B_exact = sp.dot(A_exact, perfect_fit)
    """ 添加高斯噪声到输入数据。 """
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    """ 添加高斯噪声到输出数据。 """
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)
    """ 添加异常值。 """
    if 1:
        """ 设置异常值数量。 """
        n_outliers = 100
        """ 生成所有样本的索引。 """
        all_idxs = np.arange(A_noisy.shape[0])
        """ 随机打乱索引。 """
        np.random.shuffle(all_idxs)
        """ 选择前n_outliers个索引作为异常值索引。 """
        outlier_idxs = all_idxs[:n_outliers]
        """ 替换异常值索引处的输入数据。 """
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        """ 替换异常值索引处的输出数据。 """
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))
    """ 合并输入和输出数据。 """
    all_data = np.hstack((A_noisy, B_noisy))
    """ 设置输入特征的列索引。 """
    input_columns = range(n_inputs)
    """ 设置输出特征的列索引。 """
    output_columns = [n_inputs + i for i in range(n_outputs)]
    """ 设置调试模式标志。 """
    debug = False
    """ 创建线性最小二乘模型对象。 """
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)
    """ 使用最小二乘法拟合所有数据。 """
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    """ 使用RANSAC算法拟合数据。 """
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)
    """ 绘制结果。 """
    if 1:
        """ 对精确输入数据按第一列排序。 """
        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]
        """ 绘制所有数据点。 """
        if 1:
            plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            """ 绘制RANSAC算法识别的内点。 """
            plt.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
        else:
            """ 绘制非异常值数据点。 """
            plt.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            """ 绘制异常值数据点。 """
            plt.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')
        """ 绘制RANSAC算法拟合的直线。 """
        plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
        """ 绘制精确系统对应的直线。 """
        plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='exact system')
        """ 绘制最小二乘法拟合的直线。 """
        plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit')
        """ 显示图例。 """
        plt.legend()
        """ 显示图形。 """
        plt.show()

if __name__ == "__main__":
    """ 调用test函数运行示例。 """
    test()
