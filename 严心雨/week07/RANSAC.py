import numpy as np
import scipy as sp
import scipy.linalg as sl


# 将所有的样本点随机分为2堆
def random_partition(n, n_data):
    """1.n 最少样本数
       2.n_data 所有样本点集 需是数组形式的[X,Y]"""
    """np.arange()≈Python内置的range()，允许我们创建一个数值范围内的数组
       例如np.arange(5) 返回[0 1 2 3 4]
    """
    all_index = np.arange(n_data)  # 获取下标索引 n_data=data.shape[0]
    # s随机打乱下标
    np.random.shuffle(all_index)
    index1 = all_index[:n]
    index2 = all_index[n:]
    return index1, index2


# 最小二乘求线性解，求得斜率、最小误差
class LinearLeastSquareModel(object):  # (object)可省略
    # 初始化实例变量
    def __init__(self, input_columns, output_columns, debug=False):  # debug=False?
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        """sl.lstsq(A,B)计算方程Ax=B的最小二乘解
           返回值：
           x:最小二乘解（数组），斜率
           resid:残差和
           rank:A的有效秩
           s:min(A,B) 
        """
        x, resid, rank, s = sl.lstsq(A, B)
        return x

    def get_error(self, data, model):  # model 就是fit()的结果x
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = np.dot(A, model)  # 计算的y值,B_fit = model.k*A(+ model.b)
        # 测试数据的真实值和测试数据的估计值做一个平方和便得到误差
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
        输入:
            data - 样本点 全部的数据点集
            model - 假设模型:事先自己确定（这里是斜率）
            n - 生成模型所需的最少样本点
            k - 最大迭代次数
            t - 阈值:作为判断点满足模型的条件 认为设定的用于判断误差接受许可的范围
            d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
        输出:
            bestfit - 最优拟合解（返回nil,如果未找到）

        iterations = 0
        bestfit = nil #后面更新
        besterr = something really large #后期更新besterr = thiserr
        while iterations < k
        {
            maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
            maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
            alsoinliers = emptyset #满足误差要求的样本点,开始置空
            for (每一个不是maybeinliers的样本点)
            {
                if 满足maybemodel即error < t
                    将点加入alsoinliers
            }
            if (alsoinliers样本点数目 > d)
            {
                %有了较好的模型,测试模型符合度
                bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
                thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
                if thiserr < besterr
                {
                    bestfit = bettermodel
                    besterr = thiserr
                }
            }
            iterations++
        }
        return bestfit
        """
    iterations = 0  # 循环次数
    bestfit = None  # 最优拟合结果 最优斜率
    besterr = np.inf  # 最小误差 一开始设为正无穷

    while iterations < k:
        maybe_idx, test_idx = random_partition(n, data.shape[0])
        # print('test_idx\n',test_idx)
        maybe_inliers = data[maybe_idx, :]  # maybe_idx行的所有列数据 [Xi,Yi]
        test_inliers = data[test_idx, :]
        # maybe_inliers这部分数据用于做直线拟合,直线拟合采用的是最小二乘法，得到拟合到的直线的斜率maybemodel
        maybe_model = model.fit(maybe_inliers)
        # 再用 maybemodel 值去算剩余的test_inliers（X）点 计算 test_inliers（Y）-估计值 与 真实值test_inliers（Y）-它本身的Y值 的误差平方和最小值
        test_error = model.get_error(test_inliers, maybe_model)
        print('test_error = ', test_error < t)
        # 把误差范围在t内的点归入内群点
        also_idx = test_idx[test_error < t]
        also_inliers = data[also_idx, :]  # also_idx 行的所有列数据

        if debug:
            print('test_error.min()', test_error.min())
            print('test_error.max()', test_error.max())
            print('numpy.mean(test_error)', numpy.mean(test_error))
            print('iterations %d,len(also_inliers) = %d' % (iterations, len(also_inliers)))

        if len(also_inliers) > d:
            better_inliners = np.concatenate((maybe_inliers, also_inliers))
            better_model = model.fit(better_inliners)
            better_error = model.get_error(better_inliners, better_model)
            this_error = np.mean(better_error)  # 平均误差作为新的误差
            if this_error < besterr:
                besterr = this_error
                bestfit = better_model
                best_idx_inliers = np.concatenate((maybe_idx, also_idx))
        iterations += 1

    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_idx_inliers}
    else:
        bestfit


def test():
    # 生成理想数据
    n_sample = 500  # 500个样本数
    n_inputs = 1
    n_outputs = 1

    # 随机生成500个0-20的Xi
    A_exact = 20 * np.random.random((n_sample, n_inputs))
    print('A_exact',A_exact.shape)
    # np.random.normal()专门用于生成符合正太分布（高斯分布）的随机数函数
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    # B=x*k
    B_exact = sp.dot(A_exact, perfect_fit)
    # 分别给横(X)纵(Y)坐标加上高斯噪音
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # ★ size=
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    if 1:
        # 添加局外点
        # 将其中的某些点变为outliers
        # 100个局外点
        n_outliers = 100
        # 获取索引0-499
        all_index = np.arange(A_noisy.shape[0])
        # 将all_idxs打乱
        np.random.shuffle(all_index)
        # 100个0-500的随机局外点
        outliers_index = all_index[:n_outliers]
        # 改变A_noisy ,B_noisy部分数据为局外点
        A_noisy[outliers_index] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outliers_index] = 50 * np.random.random(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi ★ size=
        # print(A_noisy[outliers_index])

        # set model

        # np.hstack() 按水平方向（列顺序）堆叠数组构成一个新的数组。堆叠的数组需要具有相同的维度
        all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....)
        input_columns = range(n_inputs)  # 0
        output_columns = [n_inputs + i for i in range(n_outputs)]  # 1
        # 最小二乘求线性解,用于RANSAC的输入模型
        debug = False
        model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)

        """
        sp.linalg.lstsq() 计算方程Ax=b的最小二乘解
        all_data[:,0]打印矩阵（所有行的）第一列
        all_data[0,:]打印矩阵（所有列）第一行
        linear_fit 最小二乘解（斜率），形状和b相同
        resids 残差
        rank 返回矩阵a的秩
        s a的奇异值 
        """
        linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

        """
        最后用RANSAC拟合出来的结果
        run RANSAC 算法
        ransac_fit 最优拟合解集合
        ransac_data 最优解的序号集合
        """
        ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

        if 1:
            import pylab
            """
            pylab.plot(x,y,marker='o',linestyle='--',markerfacecolor='r')
            x:x轴坐标列表
            y:y轴坐标列表
            marker:点型
            linestyle:连接线型
            markerfacecolor:
            """
            # np.argsort() 用于获取数组由小到大排序后的索引的函数
            sort_idxs = np.argsort(A_exact[:, 0]) #★ ≠np.argsort(A_exact)
            A_col0_sorted = A_exact[sort_idxs] # 数值是排序好的，由小到大

            if 1:
                pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
                pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                           label='ransac data')

            else:
                pylab.plot(A_noisy[non_outliers_idx, 0], B_noisy[non_outliers_idx, 0], 'k.', label='noisy data')  # ?
                pylab.plot(A_noisy[outliers_idx, 0], B_noisy[outliers_idx, 0], 'r.', label='outlier data')  # ?

            pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
            pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='perfect fit')
            pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit')

        pylab.legend()
        pylab.show()


if __name__ == '__main__':
    test()
