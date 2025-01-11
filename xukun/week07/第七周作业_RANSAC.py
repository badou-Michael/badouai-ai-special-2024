import numpy as np
import scipy.linalg as sl
import scipy as sp


def ransac(data, model, min_samples, max_iter, threshold_value, d, debug=False, return_all=False):
    # 设置初始迭代次数
    # RANSAC算法实现，目标是从数据中找到最好的模型拟合
    iterationsc = 0
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    while iterationsc < max_iter:
        maybe_idxs, test_idxs = random_partition(min_samples, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < threshold_value]
        also_inliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterationsc, len(also_inliers)))
        if len(also_inliers) > d:
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            better_err = model.get_error(betterdata, bettermodel)
            this_err = np.mean(better_err)
            if this_err < besterr:
                bestfit = bettermodel
                besterr = this_err
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterationsc += 1
    if bestfit is None:
        raise ValueError('did not meet fit acceptance criteria')
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(min_samples, min_samples_data):
    # 随机划分数据为两部分
    # 返回满足最小样本数量的随机索引
    all_idxs = np.arange(min_samples_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:min_samples]
    idxs2 = all_idxs[min_samples:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False):
        # 初始化线性最小二乘模型的输入和输出列
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # 拟合线性模型
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, res, rank, s = sl.lstsq(A, B)
        return x

    def get_error(self, data, model):
        # 计算模型的误差
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = np.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def test():
    # 测试函数，生成数据并测试RANSAC算法
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = np.dot(A_exact, perfect_fit)

    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)
    if 1:
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        B_noisy[outlier_idxs] = 60 * np.random.normal(size=(n_outliers, n_outputs))
    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug, True)
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
