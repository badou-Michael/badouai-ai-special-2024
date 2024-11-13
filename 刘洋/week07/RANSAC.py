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
        bestfit - 最优拟合解（返回nil,如果未找到）/斜率

        iterations:迭代次数
    """
    iterations = 0
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    while iterations < k:
        # maybe_idxs内群点索引，test_idxs待判断的内群点索引
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        print('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]    # 内群点数据
        test_points = data[test_idxs]          # 待判断的内群点数据
        maybemodel = model.fit(maybe_inliers)  # 根据内群点拟合模型，返回斜率

        # 判断非内群点是否为内群点
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        print('test_err = ', test_err < t)
        also_idxs = test_idxs[test_err < t]
        print('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :]

        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        # if len(also_inliers > d):
        print('d = ', d)

        # 停止条件，至少d个样本点
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 加入新的内群点
            bettermodel = model.fit(betterdata)                         # 重新拟合
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 加入新的内群点索引
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    all_idxs = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_idxs)   # 打乱索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组/取出某一列
        A = np.vstack([data[:, i] for i in self.input_columns]).T   # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B)
        return x  # 返回最小平方和向量/斜率

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T   # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = np.dot(A, model)                                    # 计算拟合后的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)            # 拟合值与真实值差方和
        return err_per_point


def test():
    # 确定一组线性数据
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # x
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机斜率k
    B_exact = np.dot(A_exact, perfect_fit)                  # y = xk

    # 加入高斯噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 加入噪声后的Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 加入噪声后的Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 索引0-499
        np.random.shuffle(all_idxs)             # 打乱索引
        outlier_idxs = all_idxs[:n_outliers]    # 取100个"局外点"
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))        # 噪声+"局外点"的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 噪声+"局外点"的Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) 400加入噪声的线性点+100"局外点" (500,2)
    input_columns = range(n_inputs)                            # 数组的第一列x:0   range(0,1)
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1   [1]
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型

    # 直接最小二乘法求出拟合曲线效率
    # a(array_like, shape(M, N)): 一个M×N的矩阵，表示线性方程组的系数矩阵。M是方程的数量，N是变量的数量。
    # b(array_like, shape(M, ) or (M, K)): 这是线性方程组的右侧常数向量。
    # 如果b是一个一维数组（形状为(M, )），则求解的是一个M个方程的系统；
    # 如果b是一个二维数组（形状为(M, K)），则求解的是K个不同的右侧常数向量。
    dataA = all_data[:, input_columns]
    dataB = all_data[:, output_columns]
    linear_fit, resids, rank, s = sp.linalg.lstsq(dataA, dataB)

    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    if 1:
        import pylab 

        sort_idxs = np.argsort(A_exact[:, 0])  # 返回数组元素排序后的索引
        A_col0_sorted = A_exact[sort_idxs]     # 排序后的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")   # 最终的内群点
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        xdata = A_col0_sorted[:, 0]
        ydata = np.dot(A_col0_sorted, ransac_fit)[:, 0]
        pylab.plot(xdata,
                   ydata,
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
