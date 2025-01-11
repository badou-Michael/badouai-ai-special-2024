import numpy as np
import scipy as sp
import scipy.linalg as sl


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
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
        debug：一个布尔值，用于控制是否开启调试模式。如果为True，则会在每次迭代中打印一些关于误差的信息。
        return_all：一个布尔值，用于控制函数的返回值。如果为True，则除了返回最佳拟合模型外，还会返
                回一个包含内点索引的字典；如果为False，则只返回最佳拟合模型。

    iterations = 0
    bestfit = nil #后面更新
    besterr = something really large #后期更新besterr = thiserr
    while iterations < k 
    {
        maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
        maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
        alsoinliers = emptyset #满足误差要求的样本点,开始置空
        for (每一个不是maybeinliers的样本点)
        {
            if 满足maybemodel即error < t
                将点加入alsoinliers 
        }
        if (alsoinliers样本点数目 > d) 
        {
            %有了较好的模型,测试模型符合度
            bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
            thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
            if thiserr < besterr
            {
                bestfit = bettermodel
                besterr = thiserr
            }
        }
        iterations++
    }
    return bestfit
    """
    # 初始化迭代次数为 0
    iterations = 0
    # 初始化最佳拟合模型为 None
    bestfit = None
    # 设置初始最小误差为无穷大
    besterr = np.inf
    # 初始化最佳内点索引为 None
    best_inlier_idxs = None
    # 开始迭代，直到达到最大迭代次数 k
    while iterations < k:
        # 随机划分数据，得到可能内点的索引和测试点的索引
        maybe_idxs, test_idxs = random_partition(n, data.shape[0]) #第一个参数n是要划分出的一部分索引的数量，第二个参数是数据的总长度
        print('test_idxs = ', test_idxs)
        # 根据索引提取可能内点数据
        maybe_inliers = data[maybe_idxs, :]
        # 根据索引提取测试点数据
        test_points = data[test_idxs]
        # 使用可能内点数据拟合一个临时模型
        maybemodel = model.fit(maybe_inliers)
        # 计算测试点在临时模型下的误差
        test_err = model.get_error(test_points, maybemodel)
        print('test_err = ', test_err < t)
        # 从测试点中选择误差小于阈值 t 的点的索引
        also_idxs = test_idxs[test_err < t]
        print('also_idxs = ', also_idxs)
        # 根据索引提取新的可能内点数据
        also_inliers = data[also_idxs, :]
        if debug:
            # 如果开启调试模式，打印一些信息
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        print('d = ', d)
        if (len(also_inliers) > d):
            # 如果新的可能内点数量大于阈值 d
            # 将最初的可能内点和新的可能内点连接起来
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            # 使用新的数据重新拟合一个更好的模型
            bettermodel = model.fit(betterdata)
            # 计算新数据在更好模型下的误差.两个参数，第一个参数是要计算误差的数据点，第二个参数是已经拟合好的模型。
            better_errs = model.get_error(betterdata, bettermodel)
            # 计算新的平均误差
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                # 如果新的平均误差小于当前最小误差
                # 更新最佳拟合模型和最小误差
                bestfit = bettermodel
                besterr = thiserr
                # 更新最佳内点索引
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        # 增加迭代次数
        iterations += 1
    if bestfit is None:
        # 如果在所有迭代结束后仍然没有找到合适的拟合模型，抛出异常
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        # 如果要求返回所有信息，返回最佳拟合模型和包含内点索引的字典
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        # 否则，只返回最佳拟合模型
        return bestfit


def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    # 获取从 0 到 n_data - 1 的索引数组
    all_idxs = np.arange(n_data)
    # 随机打乱索引数组
    np.random.shuffle(all_idxs)
    # 取前 n 个索引作为第一部分
    idxs1 = all_idxs[:n]
    # 取 n 之后的索引作为第二部分
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class LinearLeastSquareModel:
    # 最小二乘求线性解,用于 RANSAC 的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack 按垂直方向（行顺序）堆叠数组构成一个新的数组
        # 对于输入列，从数据中提取对应列并转置成列向量后堆叠在一起形成矩阵 A
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        # 对于输出列，同样操作形成矩阵 B
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 使用 scipy.linalg.lstsq 函数进行最小二乘拟合，返回拟合参数 x、残差 resids、矩阵 A 的秩 rank 和奇异值 s
        x, resids, rank, s = sl.lstsq(A, B)
        return x  # 返回最小二乘拟合得到的参数向量

    def get_error(self, data, model):
        # 与 fit 方法类似，构建输入矩阵 A
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        # 构建输出矩阵 B
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 使用模型参数 model 和输入矩阵 A 进行计算得到拟合的输出值 B_fit
        B_fit = sp.dot(A, model)
        # 计算每行的误差平方和，即真实输出 B 和拟合输出 B_fit 的差的平方和
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    # 随机生成 0 - 20 之间的 500 个数据作为输入数据 A_exact
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    # 随机生成一个斜率作为 perfect_fit，形状为 (n_inputs, n_outputs)
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    # 通过矩阵乘法生成理想的输出数据 B_exact，相当于 y = x * k
    B_exact = sp.dot(A_exact, perfect_fit)

    # 加入高斯噪声
    # 在理想输入数据 A_exact 上加入高斯噪声得到 A_noisy，代表带噪声的输入数据 Xi
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    # 在理想输出数据 B_exact 上加入高斯噪声得到 B_noisy，代表带噪声的输出数据 Yi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    if 1:
        # 添加"局外点"
        n_outliers = 100
        # 获取索引 0 - 499
        all_idxs = np.arange(A_noisy.shape[0])
        # 随机打乱索引
        np.random.shuffle(all_idxs)
        # 取前 100 个索引作为局外点的索引
        outlier_idxs = all_idxs[:n_outliers]
        # 重新生成局外点的输入数据 Xi
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        # 重新生成局外点的输出数据 Yi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    # setup model
    # 将输入数据 A_noisy 和输出数据 B_noisy 水平拼接在一起得到 all_data
    all_data = np.hstack((A_noisy, B_noisy))
    # 设置输入列的索引为 0，表示 all_data 的第一列
    input_columns = range(n_inputs)
    # 设置输出列的索引为 1，表示 all_data 的第二列
    output_columns = [n_inputs + i for i in range(n_outputs)]
    debug = False
    # 创建线性最小二乘模型的实例 model
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)

    # 使用 scipy.linalg.lstsq 进行线性拟合
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    if 1:
        import pylab

        # 对 A_exact 的第一列进行排序，得到排序后的索引
        sort_idxs = np.argsort(A_exact[:, 0])
        # 根据排序后的索引对 A_exact 进行排序得到 A_col0_sorted
        A_col0_sorted = A_exact[sort_idxs]

        if 1:
            # 绘制带噪声的数据点，标记为黑色点 '.'，标签为 'data'
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
            # 绘制 RANSAC 算法找到的内点，标记为蓝色叉 'bx'，标签为 "RANSAC data"
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            # 如果条件不为真，绘制不带局外点的数据点，标记为黑色点 '.'，标签为 'noisy data'
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            # 绘制局外点，标记为红色点 '.'，标签为 'outlier data'
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        # 绘制 RANSAC 算法拟合的曲线，根据排序后的输入数据和拟合参数计算输出数据
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        # 绘制理想系统的曲线，根据排序后的输入数据和理想参数计算输出数据
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        # 绘制线性拟合的曲线，根据排序后的输入数据和线性拟合参数计算输出数据
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        # 显示图例
        pylab.legend()
        # 显示图形
        pylab.show()


if __name__ == "__main__":
    test()
