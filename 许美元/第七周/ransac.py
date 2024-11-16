

import numpy as np
import scipy as sp
import scipy.linalg as sl


class LinearLeastSquareModel:
    '''用于执行线性最小二乘拟合和计算误差。'''
    def __init__(self, input_columns, output_columns, debug=False):
        '''接受 input_columns 和 output_columns 参数来指定输入和输出特征的列索引，
        以及一个可选的 debug 参数用于调试。'''
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        '''用于执行线性最小二乘拟合。'''
        # np.vstack 按垂直方向（行顺序）堆叠数组构成一个新的数组.第一列Xi-->行Xi
        a = np.vstack( [data[:, i] for i in self.input_columns] ).T
        b = np.vstack( [data[:, i] for i in self.output_columns] ).T
        # 使用 SciPy 的 lstsq 函数（最小二乘法）来计算矩阵 a 和 b 之间的最佳拟合参数 x，以及其他统计信息。
        x, resids, rank, s = sl.lstsq(a, b)
        # 返回计算出的线性模型参数 x。
        return x

    def get_error(self, data, model):
        '''用于计算数据点与模型预测之间的误差。'''
        a = np.vstack( [data[:, i] for i in self.input_columns] ).T
        b = np.vstack( [data[:, i] for i in self.output_columns] ).T
        # 使用 SciPy 的 dot 函数计算模型预测值 b_fit，它是输入特征矩阵 a 和模型参数 model 的点积。
        b_fit = sp.dot(a, model)
        # 计算每个数据点的实际输出 b 和预测输出 b_fit 之间的平方误差，
        # 并对每个数据点的误差求和，得到每个点的总误差 err_per_point。
        err_per_point = np.sum( (b - b_fit)**2, axis=1 )
        return  err_per_point


def random_partition(n, n_data):
    '''
    用于将数据集随机分成两部分：一部分包含 n 个样本点，另一部分包含剩余的样本点。
    【参数】
        n 是你想要从数据集中随机选择的样本点数量;
        n_data 是数据集中样本点的总数。
    '''
    all_idxs = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_idxs)   # 打乱下标索引
    idxs1 = all_idxs[:n]  # 这些索引代表了随机选择的 n 个样本点。
    idxs2 = all_idxs[n:]  # 这些索引代表了剩余的样本点。
    return idxs1, idxs2

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    '''
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回None,如果未找到）
    '''
    iterations = 0  # 初始化迭代次数为 0。
    bestfit = None  # 初始化最佳拟合模型为 None。
    besterr = np.inf  # 初始化最佳误差为无穷大，用于后续比较。
    best_inlier_idxs = None # 初始化最佳拟合解的索引为 None。
    while iterations < k:
        # 随机分割数据为两部分：一部分用于模型拟合（maybe_idxs），另一部分用于测试（test_idxs）。
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        print('test_idxs = ', test_idxs)
        # 从数据中提取用于模型拟合的样本点。
        maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
        # 从数据中提取用于测试的样本点。
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        # 使用模型拟合函数 fit 对样本点进行拟合。
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        # 计算测试样本点的误差。
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        print('test_err = ', test_err < t)
        # 根据误差阈值 t 筛选出满足模型条件的样本点索引。
        also_idxs = test_idxs[test_err < t]
        print('also_idxs = ', also_idxs)
        # 从数据中提取满足模型条件的样本点。
        also_inliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        # if len(also_inliers > d):
        print('d = ', d)
        if (len(also_inliers) > d):
            # 将用于模型拟合的样本点和满足模型条件的样本点合并。
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            # 使用合并后的样本点重新拟合模型。
            bettermodel = model.fit(betterdata)
            # 计算合并后样本点的误差。
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
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

def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    B_exact = sp.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resids, rank, s = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

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


if __name__ == '__main__':
    test()