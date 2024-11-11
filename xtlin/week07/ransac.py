import numpy as np
import scipy as sp
import scipy.linalg as sl  # 线性代数子模块

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        print('test_idxs: ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_indxs)行数据(Xi,Yi)
        test_points = data[test_idxs]
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算误差：平方和最小
        print('test_err: ', test_err < t)
        also_idxs = test_idxs[test_err < t]
        print('also_idxs: ', also_idxs)
        also_inliers = data[also_idxs]
        if debug:
            print('test_err.min(): ', test_err.min())
            print('test_err.max(): ', test_err.max())
            print('np.mean(test_err)', np.mean(test_err))
            print('iteration %d: len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        # if len(also_inliers) > d):
        print('d: ', d)
        if len(also_inliers) > d:
            better_data = np.concatenate((maybe_inliers, also_inliers))
            better_model = model.fit(better_data)
            better_errs = model.get_error(better_data, better_model)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = better_model
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点
        iterations += 1
    if bestfit is None:
        raise ValueError("didn't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    # 最小二乘求线性解
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi -> 行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 最后一列Yi -> 行Yi
        '''
        x：包含最小二乘解的二维数组。
        residuals：残差的平方和。残差指的是真实值与观测值的差。
        rank：矩阵a的秩。
        s：矩阵a的奇异值。'''
        x, residuals, rank, s = sl.lstsq(A, B)  # lstsq = least squares最小二乘法
        return x # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi -> 行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 最后一列Yi -> 行Yi
        B_fit = sp.dot(A, model)  # 计算y值 B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


def test():
    # 生成数据
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    X_exact = 20 * np.random.random((n_samples, n_inputs))  #随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  #随机线性度，即随机生成一个斜率
    Y_exact = sp.dot(X_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    X_noisy = X_exact + np.random.normal(size=X_exact.shape)  # 500 * 1行向量,代表Xi
    Y_noisy = Y_exact + np.random.normal(size=Y_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加“局外点”
        n_outliers = 100
        all_idxs = np.arange(X_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 打乱all_idxs
        outlier_idxs = all_idxs[:n_outliers]  # 100个取自0-499的随机局外点
        X_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  #加入噪声和局外点的Xi
        Y_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  #加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((X_noisy, Y_noisy))  # 形式([Xi,Yi]...) shape:(500, 2)
    input_columns = range(n_inputs)  # 输入列；第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 输出列; 最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型

    # 普通的最小二乘法
    linear_fit, residuals, rank, s = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # RANSAC算法
    ransac_fit, ransac_data = ransac(all_data, model, n=50, k=1000, t=7e3, d=300, debug=debug, return_all=True)

    if 1:
        import pylab

        sort_idxs = np.argsort(X_exact[:, 0])
        A_col0_sorted = X_exact[sort_idxs]  # 秩为2的数组
        A_col0_sorted = X_exact

        if 1:
            pylab.plot(X_noisy[:, 0], Y_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(X_noisy[ransac_data['inliers'], 0], Y_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        # print('data_shape', np.dot(A_col0_sorted, ransac_fit).shape)  # (500, 1)
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()  # 将label转换为图例
        pylab.show()


if __name__ == '__main__':
    test()
  
