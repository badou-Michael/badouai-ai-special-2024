import numpy as np
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
    """
    iterations = 0
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None

    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]  # 获取n行数据(Xi,Yi)
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算误差
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]

        if debug:
            print(f"Iteration {iterations}, also_inliers: {len(also_inliers)}, best_err: {besterr}")

        if len(also_inliers) > d:
            betterdata = np.vstack((maybe_inliers, also_inliers))  # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点
        iterations += 1

    if bestfit is None:
        raise ValueError("Did not meet fit acceptance criteria")

    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """Return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.random.permutation(n_data)
    return all_idxs[:n], all_idxs[n:]


class LinearLeastSquareModel:
    # 最小二乘求线性解，用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, resids, rank, s = sl.lstsq(A, B)
        return x

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = A @ model  # 计算的y值
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # 每行的平方误差
        return err_per_point


def test():
    # 生成理想数据
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = A_exact @ perfect_fit

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    # 添加"局外点"
    n_outliers = 100
    all_idxs = np.random.permutation(A_noisy.shape[0])
    outlier_idxs = all_idxs[:n_outliers]
    A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
    B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    # setup model
    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = list(range(n_inputs))
    output_columns = [n_inputs + i for i in range(n_outputs)]
    model = LinearLeastSquareModel(input_columns, output_columns)

    linear_fit, resids, rank, s = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC algorithm
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, return_all=True)

    import pylab
    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]

    pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
    pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    pylab.plot(A_col0_sorted[:, 0], A_col0_sorted @ ransac_fit, label='RANSAC fit')
    pylab.plot(A_col0_sorted[:, 0], A_col0_sorted @ perfect_fit, label='exact system')
    pylab.plot(A_col0_sorted[:, 0], A_col0_sorted @ linear_fit, label='linear fit')
    pylab.legend()
    pylab.show()


if __name__ == "__main__":
    test()
