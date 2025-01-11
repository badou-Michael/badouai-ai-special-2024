import numpy
import numpy as np
import scipy as sp
import scipy.linalg as sl
import pylab


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    输入：
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）
    """
    iterations = 0  # 当前迭代次数，初始值为0
    bestfit = None  # 最优模型参数，初始值为None
    besterr = np.inf  # 最优模型的误差，初始值为无穷大
    best_inlier_idxs = None  # 最优模型对应的内点（inliers）索引，初始值为None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])  # 随机选择n个样本点作为内点候选，并返回它们的索引maybe_idxs和剩余点的索引test_idxs
        print('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]  # 从数据集中提取maybe_idxs对应的样本点
        test_points = data[test_idxs]  # 从数据集中提取test_idxs对应的样本点
        maybemodel = model.fit(maybe_inliers)  # 使用 maybe_inliers 拟合模型，得到 maybemodel
        test_err = model.get_error(test_points, maybemodel)  # 计算 test_points 在 maybemodel 下的误差，平方和最小
        print('test_err = ', test_err < t)
        also_idxs = test_idxs[test_err < t]  # 筛选出误差小于阈值 t 的样本点索引
        print('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :]  # 从数据集中提取 also_idxs 对应的样本点
        if debug:
            # 打印误差的最小值、最大值、均值以及当前迭代中内点的数量
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        print('d = ', d)
        if len(also_inliers) > d:
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 将 maybe_inliers 和 also_inliers 合并为 betterdata
            bettermodel = model.fit(betterdata)  # 使用 betterdata 重新拟合模型，得到 bettermodel
            better_errs = model.get_error(betterdata, bettermodel)  # 计算 betterdata 在 bettermodel 下的误差 better_errs
            thiserr = np.mean(better_errs)  # 计算平均误差 thiserr
            if thiserr < besterr:
                # 更新最优模型 bestfit、最优误差 besterr 和最优内点索引 best_inlier_idx
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")  # 如果没有找到最优模型，抛出异常
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}  # 返回最优模型和内点索引
    else:
        return bestfit  # 仅返回最优模型


def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_idxs)  # 打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


# 最小二乘求线性解,用于RANSAC的输入模型
class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, residues, rank, s = sl.lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = np.dot(A, model)  # 计算的y值, B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # 计算两个数组 B 和 B_fit 之间的逐点误差平方和 sum squared error per row
        return err_per_point


def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数

    # 生成理想数据
    # 随机生成0-20之间的500个数据:行向量，生成的是均匀分布的随机数。这些数在 [0, 1) 区间内均匀分布。
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    # 随机线性度，即随机生成一个斜率（生成的是正态分布（高斯分布）的随机数。这些数围绕指定的均值（mean）和标准差（std）分布）
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k, 用scipy库中的dot函数来计算两个矩阵或向量的点积（内积）

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    # 添加"局外点"
    if 1:
        n_outliers = 100
        all_ids = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_ids)  # 将all_ids打乱
        outliers_ids = all_ids[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outliers_ids] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outliers_ids] = 50 * np.random.normal(size=(n_outliers, n_inputs))  # 加入噪声和局外点的Yi

    # setup model, 模型定义

    # hstack：用于水平堆叠（即沿第二轴，即列方向）多个数组，如果输入数组是一维的，则它们会被拼接成一个一维数组，如果输入数组是二维的，则它们会在列方向上进行拼接
    # # 一维数组
    # a = np.array([1, 2, 3])
    # b = np.array([4, 5, 6])
    # result = np.hstack((a, b))
    # print(result)  # 输出: [1 2 3 4 5 6]
    #
    # # 二维数组
    # a = np.array([[1, 2], [3, 4]])
    # b = np.array([[5, 6], [7, 8]])
    # result = np.hstack((a, b))
    # print(result)  # 输出: [[1 2 5 6]
    # #        [3 4 7 8]]

    # stack：用于沿着新轴堆叠一系列数组，从而增加一个新的维度
    # # 一维数组
    # a = np.array([1, 2, 3])
    # b = np.array([4, 5, 6])
    # result = np.stack((a, b))
    # print(result)  # 输出: [[1 2 3]
    # #        [4 5 6]]
    #
    # # 二维数组
    # a = np.array([[1, 2], [3, 4]])
    # b = np.array([[5, 6], [7, 8]])
    # result = np.stack((a, b))
    # print(result)  # 输出: [[[1 2]
    # #        [3 4]]
    # #       [[5 6]
    # #        [7 8]]]

    # 堆叠方式:
    # np.hstack
    # 沿着现有的轴（通常是第二轴）进行水平堆叠。
    # np.stack
    # 沿着新轴进行堆叠，增加了一个新的维度。
    # 输入要求:
    # np.hstack
    # 可以处理不同形状的数组，只要它们在堆叠的方向上是兼容的。
    # np.stack
    # 要求所有输入数组具有相同的形状。
    # 输出形状:
    # np.hstack
    # 的输出形状取决于输入数组的形状和堆叠方式。
    # np.stack
    # 的输出形状比输入数组多一个维度。

    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列（将输入变量和输出变量合并成一个二维数组）
    input_columns = list(range(n_inputs))  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型

    # RANSAC算法应用
    # 使用最小二乘法拟合数据
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    # 使用RANSAC算法拟合数据，并返回最优模型和内点索引
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    # 结果可视化
    if 1:
        sort_ids = np.argsort(A_exact[:, 0])
        """
        A_exact = np.array([[3, 2, 1],
                    [1, 4, 5],
                    [2, 6, 7],
                    [4, 8, 9]])
        A_exact[:, 0] 将是 [3, 1, 2, 4]
        np.argsort(A_exact[:, 0]) 将返回 [1, 2, 0, 3] np.argsort返回的是一个数组，其中的元素是原数组中元素的索引，按升序排列
        1是最小的，索引为 1
        2是第二小的，索引为 2
        3是第三小的，索引为 0
        4是最大的，索引为 3
        """
        # sort_ids 是一个一维数组，包含 A_exact 第一列的排序索引。A_exact[sort_ids] 使用这些索引来重新排列 A_exact 的行。 秩为2的数组
        A_col_sorted = A_exact[sort_ids]

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图,所有数据点
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")  # RANSAC算法识别的内点
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')  # 非局外点数据
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label="outlier data")  # 局外点数据
        pylab.plot(A_col_sorted[:, 0], np.dot(A_col_sorted, ransac_fit)[:, 0], label='RANSAC fit')  # RANSAC拟合结果
        pylab.plot(A_col_sorted[:, 0], np.dot(A_col_sorted, perfect_fit)[:, 0], label='exact system')  # 精确系统
        pylab.plot(A_col_sorted[:, 0], np.dot(A_col_sorted, linear_fit)[:, 0], label='linear fit')  # 最小二乘拟合结果
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    test()
