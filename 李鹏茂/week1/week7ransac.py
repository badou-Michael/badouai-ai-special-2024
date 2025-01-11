import numpy as np
import scipy as sp
import scipy.linalg as sl


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = sp.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    RANSAC算法：寻找最优拟合模型
    输入:
        data - 样本点（例如二维数组，每行一个样本点）
        model - 假设模型（具有fit和evaluate方法的模型）
        n - 用于生成模型的最小样本点数
        k - 最大迭代次数
        t - 阈值：判断点是否符合当前模型的误差
        d - 拟合较好时需要的最少样本点数
        debug - 是否输出调试信息，默认为 False
        return_all - 是否返回所有内点的信息，默认为 False
    输出:
        bestfit - 最优拟合解（如果未找到则返回 None）
        额外输出（如果 return_all=True）:
        - inliers: 选出的内点索引
    """

    bestfit = None  # 初始化最优拟合解
    best_inlier_count = 0  # 最佳内点数
    best_model_params = None  # 最佳模型参数
    best_inliers = []  # 最佳内点集合
    iterations = 0  # 当前迭代次数

    while iterations < k:
        # 1. 随机选择n个样本点
        sample_idxs = np.random.choice(len(data), size=n, replace=False)
        sample_data = data[sample_idxs]

        # 2. 使用这些点拟合模型
        model.fit(sample_data)

        # 3. 评估所有点与当前模型的符合程度
        inliers = []
        for i in range(len(data)):
            if model.evaluate(data[i]) < t:  # 如果该点符合当前模型
                inliers.append(i)

        # 4. 判断当前模型的内点数是否达到阈值
        inlier_count = len(inliers)

        # 输出调试信息
        if debug:
            print(f"Iteration {iterations + 1}/{k}")
            print(f"Sample indices: {sample_idxs}")
            print(f"Number of inliers: {inlier_count}")
            print(f"Best inlier count so far: {best_inlier_count}")
            print(f"Inliers indices: {inliers}")

        if inlier_count > best_inlier_count and inlier_count >= d:
            # 如果当前模型的内点数更好，更新最佳模型
            best_inlier_count = inlier_count
            best_model_params = model.get_params()  # 获取模型参数
            bestfit = model  # 设置最优拟合模型
            best_inliers = inliers  # 记录当前的内点

            if debug:
                print("New best model found!")
                print(f"Best inlier count: {best_inlier_count}")
                print(f"Best model params: {best_model_params}")

        iterations += 1

    if debug:
        if bestfit is None:
            print("No good model found after maximum iterations.")
        else:
            print("Best model found.")

    # 如果返回所有内点
    if return_all:
        return bestfit, {'inliers': best_inliers}

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

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    #
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    # Now you can access the inliers
    inliers = ransac_data['inliers']
    print("Inliers:", inliers)


if __name__ == '__main__':

    test()