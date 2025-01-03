import numpy as np
import pylab
import scipy as sp
import scipy.linalg as sl
from partd import numpy
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

#层次聚类

# X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
'''
层次聚类函数 linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。
'''
# Z = linkage(X, 'ward')
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''
# f = fcluster(Z,4,'distance')
# fig = plt.figure(figsize=(5, 3))
# dn = dendrogram(Z)
# print(Z)
# plt.show()

# RANSAC随机采样一致性

# 输入:
#     data - 样本点
#     model - 假设模型:事先自己确定
#     n - 生成模型所需的最少样本点
#     k - 最大迭代次数
#     t - 阈值:作为判断点满足模型的条件
#     d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
def ransac(data, model, n, k, t, d, flag=False, return_all=False):
    #iterations：初始化为 0，表示迭代次数。
    iterations = 0
    #bestfit：保存当前最佳拟合模型，初始为 None
    bestfit = None
    #besterr：当前最佳误差，初始化为正无穷大，表示最开始的误差是未知的
    besterr = np.inf  # 设置默认值
    #best_inlier_idxs：最佳内群索引，也初始化为 None
    best_inlier_idxs = None
    #进入迭代循环，k 是最大迭代次数
    while iterations < k:
        #random_partition(n, data.shape[0])：从一个给定的数据集 n_data 中随机选取 n 个数据点
        #maybe_idxs：表示从数据集中随机选出的 n 个样本的索引
        #test_idxs：表示剩余的 n_data - n 个样本的索引
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        print('test_idxs = ', test_idxs)
        #maybe_inliers：随机选取的样本，这些点将用来拟合模型
        maybe_inliers = data[maybe_idxs, :]
        #test_points：另一部分数据，用来测试模型的拟合效果
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        #线性回归
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        #计算模型对 test_points 测试数据的误差。model.get_error() 会返回一个误差值数组
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        print('test_err = ', test_err < t)
        # 将误差小于阈值的点的索引提取出来
        also_idxs = test_idxs[test_err < t]
        print('also_idxs = ', also_idxs)
        # 根据also_idxs小于阈值的索引对应值拿出来放入also_inliers
        also_inliers = data[also_idxs, :]
        # debug日志输出
        if flag:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        print('d = ', d)
        # 验证模型效果，如果符合误差阈值的结果数量大于结果阈值
        if (len(also_inliers) > d):
            # 将 maybe_inliers 和 also_inliers 合并为一个更大的数据集
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            # 继续线性拟合
            bettermodel = model.fit(betterdata)
            # 继续根据bettermodel获取betterdata的误差值
            better_errs = model.get_error(betterdata, bettermodel)
            # 疑问：为什么这里就需要平均误差，上面的就不用
            # 答：这里是为了验证模型的误差是否符合设定的最佳误差，所以用的是平均误差
            thiserr = np.mean(better_errs)
            # 用平均后的误差与设定的最佳误差进行对比，如果小于最佳误差
            if thiserr < besterr:
                # bettermodel作为最佳拟合模型
                bestfit = bettermodel
                # thiserr作为最佳误差
                besterr = thiserr
                # 将当前选出来的内群索引与外群中符合阈值的索引合并为最佳内群索引
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
        # 迭代计数器递增
        iterations += 1
    # 如果最佳拟合模型为none,抛出异常
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    # 返回结果
    if return_all:
        # 如果 return_all 为 True，则返回最优模型 bestfit 和一个字典，字典中包含最佳局内点的索引 best_inlier_idxs
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        # 如果 return_all 为 False，只返回最优模型 bestfit
        return bestfit

def random_partition(n, n_data):
    # np.arange(n_data)：生成一个包含 n_data 个整数的数组，数组的内容是从 0 到 n_data-1 的连续整数
    all_idxs = np.arange(n_data)
    # 对 all_idxs 数组中的元素进行随机打乱
    np.random.shuffle(all_idxs)
    # 从打乱后的 all_idxs 数组中取前 n 个元素, 内群
    idxs1 = all_idxs[:n]
    # 从打乱后的 all_idxs 数组中取第 n 到最后一个元素，外群
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    #初始化
    def __init__(self, input_columns, output_columns, flag=False):
        # 用于回归的输入特征列（通常是自变量）
        self.input_columns = input_columns
        # 输出目标列（通常是因变量）
        self.output_columns = output_columns
        self.flag = flag

    #拟合函数
    def fit(self, data):
        # 自变量矩阵（输入数据）, 循环input_columns，获取data第i列，
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # sl.lstsq()最小二乘法函数
        # x：最优解，即最小二乘法的参数（回归系数,包括截距和斜率等）
        # resids：残差平方和，resids 越小结果越好
        # rank：矩阵 A 的秩，如果矩阵秩小于列数，则说明 A 是退化的（即有依赖关系），解可能不是唯一的。
        # s：奇异值分解的奇异值（可以用来判断矩阵的条件数，衡量矩阵的可逆性），较小的奇异值可能表明矩阵 A 接近奇异，解的稳定性较差。
        x, resids, rank, s = sl.lstsq(A, B)
        return x  # x 是回归系数（包括截距和斜率等）

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = sp.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def test():
#------------造数据：开始---------------#
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    # 随机生成0-20之间的500个数据:行向量
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    # 随机线性度，即随机生成一个斜率
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    # y = x * k， 将A_exact随机矩阵的值都乘以斜率k，最后生成B_exact矩阵，使得B_exact与A_exact具有线性关系
    B_exact = sp.dot(A_exact, perfect_fit)

    # 加入高斯噪声,最小二乘能很好的处理，np.random.normal()用于生成服从正态分布的随机数
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    # 定义局外点的个数
    n_outliers = 100
    # 获取高斯噪声后的行索引0-499
    all_idxs = np.arange(A_noisy.shape[0])
    # 将all_idxs打乱
    np.random.shuffle(all_idxs)
    # 选择前100个索引作为局外点
    outlier_idxs = all_idxs[:n_outliers]
    # np.random.random生成行数为n_outliers，列数为n_inputs，值为(0，1]，*20则数据都在(0,20]，替换行为outlier_idxs的A_noisy的行
    A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
    # np.random.normal生成服从正太分布的随机数矩阵*50，替换index为outlier_idxs的行
    B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))
#------------造数据：结束---------------#
    # np.hstack() 合并输入数组
    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]
    flag = False
    # 类的实例化:用最小二乘生成已知模型
    model = LinearLeastSquareModel(input_columns, output_columns, flag=flag)

    # linear_fit: 拟合的参数（即回归系数）。
    # resids: 残差（拟合数据与真实数据的误差）。
    # rank: 数据的秩。
    # s: 奇异值分解的奇异值。
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, flag=flag, return_all=True)


    sort_idxs = np.argsort(A_exact[:, 0])
    # 秩为2的数组
    A_col0_sorted = A_exact[sort_idxs]

    if 1:
        pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
        pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                   label="RANSAC data")
    else:
        pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
        pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, ransac_fit)[:, 0],
               label='RANSAC fit')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, perfect_fit)[:, 0],
               label='exact system')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, linear_fit)[:, 0],
               label='linear fit')
    pylab.legend()
    pylab.show()


if __name__ == "__main__":
    test()
