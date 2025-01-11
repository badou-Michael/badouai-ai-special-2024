import numpy as np
import scipy as sp
import scipy.linalg as sl

# 设置参数
num_least = 100
max_iter = 2000
criteria = 1e4
num_threshold = 300


def random_partition(n, n_data):
    """打乱数据并切分"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    """最小二乘法模型"""

    def __init__(self, in_cols, out_cols, debug=False):
        self.in_cols = in_cols
        self.out_cols = out_cols
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.in_cols]).T
        B = np.vstack([data[:, i] for i in self.out_cols]).T
        x, _, _, _ = sl.lstsq(A, B)
        return x

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.in_cols]).T
        B = np.vstack([data[:, i] for i in self.out_cols]).T
        B_fit = np.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def ransac(
    data,
    model,
    num_least,
    max_iter,
    criteria,
    num_threshold,
    debug=False,
    return_all=False,
):
    """ransac算法实现"""
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None

    if data.shape[0] < num_least:
        raise ValueError(f"数据点数({data.shape[0]})少于所需最小点数({num_least})")

    while iterations < max_iter:
        candidate_idxs, test_idxs = random_partition(num_least, data.shape[0])

        # 获取行数据
        candidate_inliers = data[candidate_idxs, :]
        test_points = data[test_idxs]

        # 拟合模型
        try:
            candidate_model = model.fit(candidate_inliers)
            print("test_idxs = ", test_idxs)
            test_err = model.get_error(test_points, candidate_model)
            print("test_err = ", test_err < criteria)
            add_idxs = test_idxs[test_err < criteria]
            add_inliers = data[add_idxs, :]
            print("also_idxs = ", add_idxs)

            if len(add_inliers) > num_threshold:
                betterdata = np.concatenate((candidate_inliers, add_inliers))
                bettermodel = model.fit(betterdata)
                better_errs = model.get_error(betterdata, bettermodel)
                thiserr = np.mean(better_errs)

                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr
                    best_inlier_idxs = np.concatenate((candidate_idxs, add_idxs))

                if debug:
                    print("test_err.min()", test_err.min())
                    print("test_err.max()", test_err.max())
                    print("numpy.mean(test_err)", np.mean(test_err))
                    print(
                        "iteration %d:len(alsoinliers) = %d"
                        % (iterations, len(add_inliers))
                    )

        except Exception as e:
            print(f"第 {iterations} 次迭代出现错误: {str(e)}")
            continue
        iterations += 1

    # 错误处理
    if bestfit is None or best_inlier_idxs is None:
        raise ValueError(f"在 {max_iter} 次迭代中未找到符合条件的模型。")

    if return_all:
        return bestfit, {
            "inliers": best_inlier_idxs,
            "error": besterr,
            "iterations": iterations,
        }
    else:
        return bestfit


def test():

    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random(
        (n_samples, n_inputs)
    )  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(
        size=(n_inputs, n_outputs)
    )  # 随机线性度，即随机生成一个斜率
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random(
            (n_outliers, n_inputs)
        )  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(
            size=(n_outliers, n_outputs)
        )  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(
        input_columns, output_columns, debug=debug
    )  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resids, rank, s = sp.linalg.lstsq(
        all_data[:, input_columns], all_data[:, output_columns]
    )

    ransac_fit, ransac_data = ransac(
        all_data,
        model,
        num_least,
        max_iter,
        criteria,
        num_threshold,
        debug=debug,
        return_all=True,
    )

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], "k.", label="data")  # 散点图
            pylab.plot(
                A_noisy[ransac_data["inliers"], 0],
                B_noisy[ransac_data["inliers"], 0],
                "bx",
                label="RANSAC data",
            )
        # else:
        #     pylab.plot(
        #         A_noisy[non_outlier_idxs, 0],
        #         B_noisy[non_outlier_idxs, 0],
        #         "k.",
        #         label="noisy data",
        #     )
        #     pylab.plot(
        #         A_noisy[outlier_idxs, 0],
        #         B_noisy[outlier_idxs, 0],
        #         "r.",
        #         label="outlier data",
        #     )

        pylab.plot(
            A_col0_sorted[:, 0],
            np.dot(A_col0_sorted, ransac_fit)[:, 0],
            label="RANSAC fit",
        )
        pylab.plot(
            A_col0_sorted[:, 0],
            np.dot(A_col0_sorted, perfect_fit)[:, 0],
            label="exact system",
        )
        pylab.plot(
            A_col0_sorted[:, 0],
            np.dot(A_col0_sorted, linear_fit)[:, 0],
            label="linear fit",
        )
        pylab.legend()
        pylab.show()


if __name__ == "__main__":

    test()
