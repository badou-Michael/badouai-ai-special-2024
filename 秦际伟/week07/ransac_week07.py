import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt

class LinearLeastSquareModel:
    """
    最小二乘求线性解，用于RANSAC的输入模型。
    
    :param input_columns: 输入变量的列索引
    :param output_columns: 输出变量的列索引
    :param debug: 是否开启调试模式
    """
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        """
        使用最小二乘法拟合模型。
        
        :param data: 输入数据
        :return: 拟合的模型参数
        """
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, resids, rank, s = sl.lstsq(A, B)
        return x

    def get_error(self, data, model):
        """
        计算数据点与模型之间的误差。
        
        :param data: 输入数据
        :param model: 模型参数
        :return: 每个数据点的误差
        """
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = np.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point

def random_partition(n, n_data):
    """
    随机划分数据集。
    
    :param n: 选取的样本数量
    :param n_data: 数据集的总样本数量
    :return: 两个索引数组，分别表示选取的样本和剩余的样本
    """
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

def ransac(data, model, min_samples, max_iterations, threshold, d, debug=False, return_all=False):
    """
    实现RANSAC算法。
    
    :param data: 输入数据
    :param model: 模型对象
    :param min_samples: 每次迭代中选取的最小样本数量
    :param max_iterations: 最大迭代次数
    :param threshold: 误差阈值
    :param d: 内点数量阈值
    :param debug: 是否开启调试模式
    :param return_all: 是否返回所有结果
    :return: 最佳模型及其内点索引
    """
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    iterations = 0

    while iterations < max_iterations:
        maybe_idxs, test_idxs = random_partition(min_samples, data.shape[0])
        maybe_inliers = data[maybe_idxs]
        test_points = data[test_idxs]

        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < threshold]
        also_inliers = data[also_idxs]

        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d: len(also_inliers) = %d' % (iterations, len(also_inliers)))

        if len(also_inliers) > d:
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)

            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))

        iterations += 1

    if bestfit is None:
        raise ValueError("didn't meet fit acceptance criteria")

    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit

def generate_data(n_samples, n_inputs, n_outputs, n_outliers=0):
    """
    生成带有噪声和局外点的数据集。
    
    :param n_samples: 样本数量
    :param n_inputs: 输入变量数量
    :param n_outputs: 输出变量数量
    :param n_outliers: 局外点数量
    :return: 带有噪声和局外点的数据集，以及理想模型
    """
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = np.dot(A_exact, perfect_fit)

    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    if n_outliers > 0:
        all_idxs = np.arange(A_noisy.shape[0])
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    return A_noisy, B_noisy, perfect_fit

def plot_results(A_noisy, B_noisy, ransac_data, ransac_fit, perfect_fit, linear_fit):
    """
    绘制数据点和拟合结果。
    
    :param A_noisy: 带噪声的输入数据
    :param B_noisy: 带噪声的输出数据
    :param ransac_data: RANSAC算法的结果
    :param ransac_fit: RANSAC拟合的模型
    :param perfect_fit: 理想模型
    :param linear_fit: 最小二乘拟合的模型
    """
    sort_idxs = np.argsort(A_noisy[:, 0])
    A_col0_sorted = A_noisy[sort_idxs]

    plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
    plt.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='exact system')
    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit')
    plt.legend()
    plt.show()

def main():
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    n_outliers = 100

    A_noisy, B_noisy, perfect_fit = generate_data(n_samples, n_inputs, n_outputs, n_outliers)
    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)

    linear_fit, _, _, _ = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    plot_results(A_noisy, B_noisy, ransac_data, ransac_fit, perfect_fit, linear_fit)

if __name__ == "__main__":
    main()
