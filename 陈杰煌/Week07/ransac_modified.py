import numpy as np
import scipy.linalg as sl

def ransac(data, model, min_samples, max_iterations, threshold, min_inliers, debug=False, return_all=False):
    """
    RANSAC算法实现
    输入:
        data - 样本点
        model - 假设模型(实现fit和get_error方法)
        min_samples - 每次迭代中生成模型所需的最少样本点数(n)
        max_iterations - 最大迭代次数(k)
        threshold - 阈值，用于判断点是否属于模型内点(t)
        min_inliers - 拟合满足要求的最少内点数(d)
    输出:
        best_fit - 最优拟合模型参数
        best_inliers - (可选) 内点的索引
    """
    iterations = 0
    best_fit = None
    best_error = np.inf
    best_inlier_indices = None

    while iterations < max_iterations:
        maybe_indices, test_indices = random_partition(min_samples, data.shape[0])
        maybe_inliers = data[maybe_indices]
        test_points = data[test_indices]
        
        maybe_model = model.fit(maybe_inliers)
        test_errors = model.get_error(test_points, maybe_model)
        also_inliers_indices = test_indices[test_errors < threshold]
        also_inliers = data[also_inliers_indices]

        if debug:
            print(f"Iteration {iterations}: {len(also_inliers)} inliers")

        if len(also_inliers) > min_inliers:
            combined_data = np.vstack((maybe_inliers, also_inliers))
            better_model = model.fit(combined_data)
            combined_errors = model.get_error(combined_data, better_model)
            average_error = np.mean(combined_errors)

            if average_error < best_error:
                best_fit = better_model
                best_error = average_error
                best_inlier_indices = np.concatenate((maybe_indices, also_inliers_indices))

        iterations += 1

    if best_fit is None:
        raise ValueError("未找到满足条件的拟合模型")
    return (best_fit, {'inliers': best_inlier_indices}) if return_all else best_fit


def random_partition(n, data_size):
    """返回数据的n个随机样本和剩余的样本索引"""
    all_indices = np.arange(data_size)
    np.random.shuffle(all_indices)
    return all_indices[:n], all_indices[n:]


class LinearLeastSquaresModel:
    """最小二乘线性模型, 用于RANSAC的模型输入"""

    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    
    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, _, _, _ = sl.lstsq(A, B)
        return x
    
    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = np.dot(A, model)
        return np.sum((B - B_fit) ** 2, axis=1)


def test():
    # 生成测试数据
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = np.dot(A_exact, perfect_fit)

    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    # 添加离群点
    n_outliers = 100
    all_indices = np.arange(A_noisy.shape[0])
    np.random.shuffle(all_indices)
    outlier_indices = all_indices[:n_outliers]
    A_noisy[outlier_indices] = 20 * np.random.random((n_outliers, n_inputs))
    B_noisy[outlier_indices] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]
    model = LinearLeastSquaresModel(input_columns, output_columns)

    # 运行RANSAC算法
    ransac_fit, ransac_data = ransac(all_data, model, min_samples=50, max_iterations=1000, threshold=7e3, min_inliers=300, return_all=True)

    # 绘制结果
    import matplotlib.pyplot as plt
    sort_indices = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_indices]

    plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='Data')
    plt.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC Inliers")
    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC Fit')
    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='Exact Model')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()
