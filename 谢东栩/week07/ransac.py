import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt

# RANSAC算法的实现
def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    iterations = 0  # 初始化迭代次数
    bestfit = None  # 最佳拟合模型
    besterr = np.inf  # 最佳拟合的误差，初始化为无穷大
    best_inlier_idxs = None  # 最佳拟合的内点索引

    # 执行RANSAC算法的迭代过程
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])  # 随机分割数据集
        maybe_inliers = data[maybe_idxs, :]  # 可能的内点数据
        test_points = data[test_idxs]  # 测试数据

        maybemodel = model.fit(maybe_inliers)  # 使用内点数据拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 测试数据的误差

        # 选择误差小于阈值的点作为内点
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]

        # 如果内点的数量大于最小要求的数量，则进行更好的拟合
        if len(also_inliers) > d:
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 合并内点数据
            bettermodel = model.fit(betterdata)  # 使用合并后的内点数据拟合新模型
            better_errs = model.get_error(betterdata, bettermodel)  # 计算新的误差
            thiserr = np.mean(better_errs)  # 计算当前模型的平均误差

            # 如果当前模型的误差更小，则更新最佳模型
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新内点索引

        iterations += 1  # 增加迭代次数

    # 如果未找到合适的拟合模型，抛出异常
    if bestfit is None:
        raise ValueError("didn't meet fit acceptance criteria")

    # 返回拟合结果及所有内点
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit

# 随机分割数据集
def random_partition(n, n_data):
    all_idxs = np.arange(n_data)  # 生成所有数据的索引
    np.random.shuffle(all_idxs)  # 打乱索引顺序
    return all_idxs[:n], all_idxs[n:]  # 返回前n个和剩余的索引

# 线性最小二乘法模型
class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns  # 输入列索引
        self.output_columns = output_columns  # 输出列索引
        self.debug = debug  # 调试标志

    def fit(self, data):
        A = data[:, self.input_columns]  # 提取输入数据
        B = data[:, self.output_columns]  # 提取输出数据
        # 使用线性最小二乘法拟合模型
        x, _, _, _ = sl.lstsq(A, B)
        return x  # 返回拟合的参数

    def get_error(self, data, model):
        A = data[:, self.input_columns]  # 提取输入数据
        B = data[:, self.output_columns]  # 提取输出数据
        B_fit = np.dot(A, model)  # 计算模型的预测输出
        # 计算每个数据点的误差
        return np.sum((B - B_fit) ** 2, axis=1)

# 测试RANSAC算法
def test():
    n_samples = 500  # 样本数量
    n_inputs = 1  # 输入特征数量
    n_outputs = 1  # 输出特征数量
    X = 20 * np.random.random((n_samples, n_inputs)) - 10  # 生成输入数据，范围在[-10, 10]
    perfect_fit = 60 * np.random.random((n_inputs, n_outputs)) - 30  # 真实模型参数

    Y = np.dot(X, perfect_fit)  # 根据真实模型计算输出数据

    # 添加一些噪声到一半的数据
    X[:n_samples // 2] += np.random.normal(size=(n_samples // 2, n_inputs))
    Y[:n_samples // 2] += np.random.normal(size=(n_samples // 2, n_outputs))

    data = np.hstack((X, Y))  # 将输入和输出数据合并为一个数据集
    data[:n_samples // 5] = 20 * np.random.random((n_samples // 5, n_inputs + n_outputs)) - 10  # 添加一些离群点

    input_columns = range(n_inputs)  # 输入数据的列索引
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 输出数据的列索引
    model = LinearLeastSquareModel(input_columns, output_columns, debug=False)  # 创建模型实例

    # 使用RANSAC算法进行拟合
    ransac_fit, ransac_data = ransac(data, model, n=50, k=2000, t=5e3, d=300, return_all=True)

    # 打印真实模型和RANSAC拟合模型的参数
    print("真实模型参数：", perfect_fit.ravel())
    print("RANSAC拟合模型参数：", ransac_fit.ravel())

    # 获取内点和外点数据
    inlier_data = data[ransac_data['inliers']]
    outlier_data = np.delete(data, ransac_data['inliers'], axis=0)

    # 可视化结果
    plt.scatter(outlier_data[:, 0], outlier_data[:, 1], color='red', label='Outliers')  # 绘制离群点
    plt.scatter(inlier_data[:, 0], inlier_data[:, 1], color='green', label='Inliers')  # 绘制内点

    x_range = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)  # 创建x轴的范围
    y_range = np.dot(x_range.reshape(-1, 1), ransac_fit)  # 根据拟合模型计算y值
    plt.plot(x_range, y_range, color='blue', label='RANSAC Fit')  # 绘制RANSAC拟合直线

    plt.legend()  # 显示图例
    plt.xlabel("Input")  # 设置x轴标签
    plt.ylabel("Output")  # 设置y轴标签
    plt.title("RANSAC Results")  # 设置标题
    plt.show()  # 显示图形

if __name__ == "__main__":
    test()  # 执行测试函数
