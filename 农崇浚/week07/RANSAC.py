import numpy as np
import scipy.linalg as sl
import scipy as sp

# 1.随机抽样
def random_partition(n, n_data):
    """从数据集中随机选取 n 个样本点，返回两个索引数组，一个是被选中的样本点，另一个是剩余的测试点"""
    all_idxs = np.arange(n_data) # 生成一个包含所有样本点索引的数组
    np.random.shuffle(all_idxs) # 随机打乱这些索引
    idxs1 = all_idxs[:n] # 取前 n 个作为抽样点
    idxs2 = all_idxs[n:] # 剩下的作为测试点
    return idxs1, idxs2


#2.拟合模型
class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # 拟合线性模型，使用最小二乘法
        A = np.vstack([data[:, i] for i in self.input_columns]).T # 将输入特征放入 A
        B = np.vstack([data[:, i] for i in self.output_columns]).T # 将输出特征放入 B
        x, resids, rank, s = sl.lstsq(A, B) # 使用最小二乘法拟合
        return x

    def get_error(self, data, model):
        # 计算误差，误差是预测值和真实值之间的平方差
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = sp.dot(A, model) # 使用模型预测 B
        err_per_point = np.sum((B - B_fit) ** 2, axis=1) # 计算每个点的平方误差
        return err_per_point

# 3.RANSAC
# RANSAC 的核心部分，反复随机抽样并拟合模型，判断模型是否足够好
def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    输入:
        data - 样本点
        model - 假设模型
        n - 生成模型所需的最少样本点数
        k - 最大迭代次数
        t - 阈值，判断点是否为内点的误差阈值
        d - 当符合模型的样本点数目大于 d 时，认为模型足够好
    输出:
        bestfit - 最优拟合解（如果未找到，则返回 None）
    """
    iterations = 0
    bestfit = None
    besterr = np.inf  # 初始化一个非常大的误差
    best_inlier_idxs = None

    while iterations < k:
        # 随机选取 n 个点作为可能的内点
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]  # 选出的内点
        test_points = data[test_idxs]  # 其余的测试点

        # 拟合模型
        maybemodel = model.fit(maybe_inliers)

        # 计算测试点的误差
        test_err = model.get_error(test_points, maybemodel)

        # 判断哪些点符合该模型（误差小于阈值 t）
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]

        # 如果符合的点足够多，认为这是一个较好的模型
        if len(also_inliers) > d:
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)

            # 如果当前模型的误差小于之前的最优模型，则更新最优解
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))

        iterations += 1

    # 如果没有找到合适的模型，抛出错误
    if bestfit is None:
        raise ValueError("未找到符合要求的模型")

    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit

# 4.测试RANSAC
def test():
    # 生成理想数据
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs)) # 生成 500 个随机样本
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs)) # 随机生成模型的斜率
    B_exact = sp.dot(A_exact, perfect_fit) # 根据斜率生成对应的输出

    # 加入噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    # 加入一些离群点
    n_outliers = 100
    all_idxs = np.arange(A_noisy.shape[0])
    np.random.shuffle(all_idxs)
    outlier_idxs = all_idxs[:n_outliers]
    A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
    B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    # 将数据合并
    all_data = np.hstack((A_noisy, B_noisy))

    # 设置输入输出列
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]

    # 实例化模型
    model = LinearLeastSquareModel(input_columns, output_columns, debug=False)

    # 使用 RANSAC 拟合
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=False, return_all=True)

    # 可视化拟合效果
    import pylab
    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]

    pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
    pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
    pylab.legend()
    pylab.show()

if __name__ == "__main__":
    test()
