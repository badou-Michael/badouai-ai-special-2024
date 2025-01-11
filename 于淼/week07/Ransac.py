import numpy as np
import scipy as sp
import scipy.linalg as sl


def ransac(data, model, n, k, t, d, debug, return_all):
    '''
    :param data: 样本点，通常是一个二维数组，其中每行代表一个数据点。
    :param model: 假设模型，需要事先定义，并具有 fit 和 get_error 方法。
    :param n: 生成模型所需的最少样本点数。
    :param k: 最大迭代次数。
    :param t: 阈值，用于判断点是否满足模型。
    :param d: 拟合较好时，需要的样本点最少的个数，作为阈值。
    :param debug: 是否输出调试信息。
    :param return_all: 是否返回所有信息，包括最优拟合解和局内点索引。
    :return:
    '''
    iterations = 0;  # 初始化迭代次数
    bestfit = None;  # 最优拟合解
    besterr = np.inf  # 最优误差,设置为无穷大，用于后续比较误差
    best_inlier_idxs = None  # 局内点索引

    while iterations < k:
        # 随机选择n个数据点作为maybe_inliers的索引，其余作为test_points的索引
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])

        # 根据索引获取数据点
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idxs]

        # 使用maybe_inliers拟合模型
        maybemodel = model.fit(maybe_inliers)

        # 计算test_points相对于maybemodel的误差
        test_err = model.get_error(test_points, maybemodel)

        # 找出误差小于阈值t的点，作为also_inliers的索引
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]

        # 如果also_inliers的数量大于d，则进行更进一步的模型优化
        if len(also_inliers) > d:
            # 合并maybe_inliers和also_inliers，用于重新拟合模型
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 使用平均误差作为新的误差度量

            # 如果新误差小于当前最优误差，则更新最优解
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))

        # 迭代次数加1
        iterations += 1

        # 如果没有找到最优解，则抛出异常
    if bestfit is None:
        raise ValueError("didn't meet fit acceptance criteria")

        # 根据return_all的值决定返回内容
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


'''
功能：
    生成一个从 0 到 n_data-1 的索引数组 all_idxs。
    打乱 all_idxs。
    返回前 n 个索引作为 idxs1，剩余索引作为 idxs2。
'''


def random_partition(n, n_data):
    '''
    :param n: 要返回的随机行数。
    :param data: 数据的总行数。
    '''
    all_idxs = np.arange(n_data)  # 生成一个包含所有索引的数组
    np.random.shuffle(all_idxs)  # 随机打乱索引数组
    idxs1 = all_idxs[:n]  # 取前n个索引作为maybe_inliers的索引
    idxs2 = all_idxs[n:]  # 取剩余的索引作为test_points的索引
    return idxs1, idxs2


class LinearLeastSquareModel:
    # 通过最小二乘法拟合线性模型
    def __init__(self, input_columns, output_columns, debug=False):
        # input_columns 和 output_columns 分别是输入和输出数据的列索引；
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    # 拟合模型
    def fit(self, data):
        # np.vstack按垂直方向堆叠数组，构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # T就是转置矩阵，第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # T就是转置矩阵，第二列Xi-->行Xi
        x, resids, rank, s = sl.lstsq(A, B)  # 使用scipy.linalg.lstsq函数,求解线性最小二乘问题
        '''
            x: 最小二乘解的系数向量
            resids: 残差和的平方
            rank: 矩阵A的秩
            s: 矩阵A的奇异值（单数）
        '''
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = sp.dot(A, model)  # 利用点乘（dot）函数来计算矩阵 A 和 model 的乘积，得出模型的预测值B_fit
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # 计算实际值 B 和预测值 B_fit 之间的平方误差,评估模型的拟合效果
        return err_per_point


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

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns],
                                                  all_data[:, output_columns])  # 实用最小二乘法对全部数据进行线性拟合

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


if __name__ == "__main__":
    test()
