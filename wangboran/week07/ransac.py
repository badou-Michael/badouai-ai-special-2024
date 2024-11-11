# -*- coding: utf-8 -*-
# author: 王博然
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

def random_partition(n_part, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data) #获取n_data下标索引
    np.random.shuffle(all_idxs) #打乱下标索引
    idxs1 = all_idxs[:n_part]
    idxs2 = all_idxs[n_part:]
    return idxs1, idxs2

# 线性最小二乘模型, 用于RANSAC的输入模型
class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self, data):
        x = data[:,self.input_columns]    # sample * input_columns(=1), 注意是2维
        y = data[:,self.output_columns]
        linear_fit, residues, rank, s = lstsq(x, y)
        return linear_fit

    def get_error(self, data, model):
        x = data[:,self.input_columns]
        y = data[:,self.output_columns]   # sample * input_columns(=1), 注意是2维
        y_fit = np.dot(x, model)          # 计算的y值, y_fit = model.k*x + model.b
        # 针对每行, 计算方差和
        err_per_point = np.sum((y - y_fit)**2, axis = 1)
        return err_per_point

def ransac(data, model, n, k, t, d):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解(返回None,如果未找到）
    """
    iterations = 0
    bestfit = None
    besterr = np.inf #设置默认值, 正无穷大
    best_inlier_idxs = None  # 内群索引
    while iterations < k:
        # maybe 用来聚合, test 用于验证
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs] #获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]    #剩余行作为 测试数据点 (Xi,Yi)
        maybemodel = model.fit(maybe_inliers) #拟合模型
        test_err = model.get_error(test_points, maybemodel) #计算误差:平方和最小
        also_idxs = test_idxs[test_err < t]   # 过滤满足阈值的点
        also_inliers = data[also_idxs]
        if (len(also_inliers) > d):
            betterdata = np.concatenate( (maybe_inliers, also_inliers) ) #样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs) #平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) ) #更新局内点,将新点加入
        iterations += 1

    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    else:
        return bestfit,{'inliers':best_inlier_idxs}

def gen_data(n_sample, n_outlier, n_input, n_output):
    x_exact = 20 * np.random.random((n_sample, n_input)) # n_sample * 1, 值范围[0,20)
    #随机线性度，即随机生成一个斜率
    perfect_fit = 60 * np.random.normal(size=(n_input, n_output))  # 1 * 1
    y_exact = np.dot(x_exact, perfect_fit) # y = x * k

    # 加入高斯噪声 (实际使用)
    x_noise = x_exact + np.random.normal( size = x_exact.shape ) # n_sample * 1行向量,代表Xi
    y_noise = y_exact + np.random.normal( size = y_exact.shape )

    # 添加"局外点"
    all_idx = np.arange(x_noise.shape[0])  # 获取索引 [0, n_sample - 1]
    np.random.shuffle(all_idx)  # 打乱顺序
    outlier_idx = all_idx[:n_outlier] # 前100个乱序索引
    x_noise[outlier_idx] = 20 * np.random.random((n_outlier, n_input))   # 局外点的x
    y_noise[outlier_idx] = 50 * np.random.normal(size = (n_outlier, n_input)) # 局外点的y

    # 拼接为最终数据
    all_data = np.hstack((x_noise, y_noise))  # [[x1,y1],[x2,y2],...], n_sampe * 2
    return perfect_fit, all_data

if __name__ == '__main__':
    n_input = 1
    n_output = 1
    n_sample = 500
    n_outlier = 100
    perfect_fit, all_data = gen_data(n_sample, n_outlier, n_input, n_output)

    input_columns = range(n_input)  # 用于从 all_data 中提取 input
    output_columns = [n_input + i for i in range(n_output)]  # 用于从 all_data中提取output
    final_x = all_data[:,input_columns]  # n_sample * 1, 注意是2维
    final_y = all_data[:,output_columns] # n_sample * 1, 注意是2维

    # 拟合 - 最小二乘法
    linear_fit, residues, rank, s = lstsq(final_x, final_y)

    # RANSAC
    model = LinearLeastSquareModel(input_columns, output_columns)
    # 最小样本点50, 最大迭代数1000, 阈值6000, 新增拟合点阈值
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 6e3, 300)
    
    # 画图
    plt.plot(final_x, final_y, 'k.', label = 'data' ) #散点图
    plt.plot(final_x[ransac_data['inliers'], 0], final_y[ransac_data['inliers'], 0], 'bx', label = "RANSAC data" )

    plt.plot(final_x[:,0], np.dot(final_x,ransac_fit)[:,0], label='RANSAC fit')
    plt.plot(final_x[:,0], np.dot(final_x,perfect_fit)[:,0], label='exact system' )
    plt.plot(final_x[:,0], np.dot(final_x,linear_fit)[:,0], label='linear fit' )
    plt.legend()
    plt.show()
