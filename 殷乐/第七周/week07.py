import cv2
import scipy as sp
import scipy.linalg as sl
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = np.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


# 1.实现层次聚类：凝聚层次聚类，自下而上的
def hierarchical_clustering(X):
    # linkage(X, method=' ')
    # 第一个参数y为一个尺寸为(m,n)的二维矩阵。一共有n个样本，每个样本有m个维度。
    # 参数method =
    #   ’single’：一范数距离
    #   ’complete’：无穷范数距离
    #   ’average’：平均距离
    #   ’centroid’：二范数距离
    #   ’ward’：离差平方和距离
    # 返回值：(n-1)*4 的矩阵 Z  记录层次聚类的层次信息
    Z = linkage(X, 'ward')
    print(Z)

    # fcluster函数:从给定链接矩阵定义的层次聚类中形成平面聚类
    # 距离阈值t:允许的最大簇间距离
    # 参数 criterion ：
    #   ’inconsistent’：预设的，如果一个集群节点及其所有后代的不一致值小于或等于 t，那么它的所有叶子后代都属于同一个平面集群。当没有非单例集群满足此条件时，每个节点都被分配到自己的集群中。
    #   ’distance’：每个簇的距离不超过 t。
    #   criterion=‘maxclust’：当取这个参数时，t 表示最大分类簇数。
    #   maxclust_monocrit :
    #   monocrit :
    #   inconsistent :
    # 输出是每一个特征的类别。
    f = fcluster(Z, 2, 'distance')
    print(f)
    # dendrogram函数,绘制层次聚类图
    dn = dendrogram(Z)
    plt.show()

"""
 输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）
"""
def ransac_fun(data, model, n, k, t, d, return_all=False):
    iterations = 0
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    while iterations < k:
        # 通过选取的你和点计算拟合模型，再用你和模型计算其他剩余点的误差平方和
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        # print('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        # print('test_err = ', test_err < t)
        # 如果误差平方和小于阈值，则认为是内点
        also_idxs = test_idxs[test_err < t]
        # print('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :]

        # 拟合较好的数量点达到一定值时
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            # 如果新的误差更小，更新内容
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    # 创建一系列连续的数值
    all_idxs = np.arange(n_data) #获取n_data下标索引
    np.random.shuffle(all_idxs) #打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


# 2.实现ransac
def ransac():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    # 生成一个形状为(n_samples, n_inputs)的数组，数组中的元素都是随机浮点数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    # np.random.normal(loc=0.0, scale=1.0, size=None):生成符合正态分布（高斯分布）的随机数的函数。
    # loc：正态分布的均值（期望值）;scale：正态分布的标准差; size：生成随机数的数量。
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    print(perfect_fit)
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k  标准值

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    # 添加"局外点"
    n_outliers = 100
    print(A_noisy.shape[0])
    # 创建一系列连续的数值
    all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
    # print(all_idxs)
    # 随机打乱数组中的元素顺序
    np.random.shuffle(all_idxs)  # 将all_idxs打乱
    outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
    # print(outlier_idxs)
    A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
    print(A_noisy.shape)
    B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    # print((all_data))
    input_columns = range(n_inputs)  # 数组的第一列x:0(线性)
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1(线性)

    model = LinearLeastSquareModel(input_columns, output_columns)  # 类的实例化:用最小二乘生成已知模型

    # 参数分别是系数矩阵的数组。如AX=B中的A和B
    # 返回值linear_fit：包含最小二乘解的二维数组。
    # resids：残差的平方和。残差指的是真实值与观测值的差。
    # rank：矩阵a的秩。
    # s：矩阵a的奇异值。
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac_fun(all_data, model, 50, 1000, 7e3, 300, return_all=True)

    # 画图
    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

    plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
    plt.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")

    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='exact system')
    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit')
    plt.legend()
    plt.show()


# hierarchical_clustering([[1, 2], [1, 9], [4, 4], [1, 2], [1, 3], [2,5]])
ransac()
