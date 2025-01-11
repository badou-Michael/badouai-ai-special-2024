# RANSAC的步骤 随机采样一致性（randomsampleconsensus）
# 1.在数据中随机选择几个点设定为内群
# 2.计算适合内群的模型e.g.y=ax+b->y=2x+3y=4x+5
# 3.把其它刚才没选到的点带入刚才建立的模型中，计算是否为内群e.g.hi=2xi+3->ri  ##残差：ri=h(xi)–yi
# 4.记下内群数量
# 5.重复以上步骤
# 6.比较哪次计算中内群数量最多,内群最多的那次所建的模型就是我们所要求的解
import numpy as np
import scipy as sp
import scipy.linalg as sl
#scipy.linalg：scipy库中的线性代数模块，用于执行矩阵分解和其他线性代数操作

'''
data：样本点数据。
model：用于拟合数据的模型。
n：生成模型所需的最少样本点数。
k：最大迭代次数。
t：阈值，用于判断点是否满足模型的条件。
d：拟合较好时需要的样本点的最小数量。
debug：是否开启调试模式，默认为False。
return_all：是否返回所有数据，默认为False。
'''
def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    iterations = 0
    bestfit = None
    besterr = np.inf #设置默认值  np.inf 是 NumPy 库中的一个特殊值，代表正无穷大（positive infinity）
    best_inlier_idxs = None
    # iterations：迭代次数计数器。
    # bestfit：存储最佳拟合模型。
    # besterr：存储最佳拟合误差，初始设为无穷大。
    # best_inlier_idxs：存储最佳拟合的内点索引。
    while iterations < k:  ##开始一个循环，直到达到最大迭代次数k。
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])  ##调用random_partition函数，随机选择n个样本点用于模型拟合，其余的用于误差测试。
        #random_partition随机地将数据集分成两部分：一部分包含 n 个样本点，另一部分包含剩余的样本点。
        # 这个函数在 RANSAC（RANdom SAmple Consensus）算法中被用来随机选择样本点以拟合模型，并评估模型的拟合度。
        print('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
        #在 Python 的 NumPy 库中，[:, :] 是一种用于多维数组的切片语法，它用于选择数组的一部分元素。具体来说，[:, :] 表示选择所有的行和所有的列，也就是整个数组。
        #data[maybe_idxs, :] 的意思是从 data 数组中选择 maybe_idxs 指定的行，并且选择这些行的所有列
        #data[maybe_idxs, :] 用于选择特定的行和所有列，而 data[maybe_idxs:] 用于选择从某个索引开始到末尾的所有行。两者在数组切片中有不同的用途和结果。
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers)  # 拟合模型 ##odel 实际上是一个 LinearLeastSquareModel 类的实例，这个类实现了最小二乘法来拟合线性模型。
        # fit 方法是 LinearLeastSquareModel 类的一个方法，它接受一组样本点作为输入，并返回拟合出的模型参数。

        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        print('test_err = ', test_err < t)  #这行代码使用布尔索引从 test_idxs 中选择出那些误差小于阈值 t 的样本点的索引，并将这些索引存储在 also_idxs 中。
        also_idxs = test_idxs[test_err < t] #代码使用布尔索引从 test_idxs 中选择出那些误差小于阈值 t 的样本点的索引，并将这些索引存储在 also_idxs 中。
        print('also_idxs = ', also_idxs)

        also_inliers = data[also_idxs, :]  #这行代码使用 also_idxs 从 data 数组中选择出对应的内点数据，并将这些数据存储在 also_inliers 中。
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',numpy.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)) )  #len(also_inliers) 是满足误差要求的样本点的数量，即 also_inliers 数组的长度。
        # if len(also_inliers > d):
        print('d = ', d)  #如果满足误差阈值 t 的样本点数量超过 d，则认为模型是一个好的拟合，并且可以用这些点来重新拟合模型以获得更好的结果。
        #
        #如果内点的数量大于d，则使用这些内点和最初随机选择的点重新拟合模型，并更新最佳模型。
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接  代码将这两组样本点合并成一个新的数组 betterdata，这个数组包含了所有可能的内点
            bettermodel = model.fit(betterdata) #使用 betterdata 来拟合一个更精确的模型，并将拟合结果存储在 bettermodel 中
            better_errs = model.get_error(betterdata, bettermodel)  ##model.get_error 是模型类的 get_error 方法，用于计算模型预测值和实际值之间的误差
            # 。这行代码计算 bettermodel 在 betterdata 上的误差，并将结果存储在 better_errs 中。
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr:  #用于检查新计算的平均误差 thiserr 是否小于当前记录的最小误差 besterr。
                #thiserr 小于 besterr，则说明新拟合的模型 bettermodel 比之前的模型有更好的拟合度。
                bestfit = bettermodel
                #如果新模型的拟合度更好，这行代码将 bettermodel 赋值给 bestfit，更新最佳拟合模型
                besterr = thiserr #同时，这行代码也将 thiserr 赋值给 besterr，更新记录的最小误差
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
                # 这行代码将初步选取的样本点索引 maybe_idxs 和新发现的内点索引 also_idxs 合并，更新最佳内点的索引集合 best_inlier_idxs。
        iterations += 1 #增加迭代次数
    if bestfit is None:  #如果 bestfit 仍然是 None，即在整个迭代过程中没有找到任何合适的模型，则可能需要抛出一个异常或返回一个错误信息，表示算法未能找到满足条件的模型。
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
            return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_idxs)  # 打乱下标索引
    idxs1 = all_idxs[:n] #这行代码通过切片操作从打乱后的索引数组 all_idxs 中选择前 n 个索引。idxs1 包含了随机选择的 n 个样本点的索引。
    idxs2 = all_idxs[n:] #idxs2 包含了剩余的样本点的索引。
    return idxs1, idxs2 #idxs1 用于拟合初始模型。dxs2 用于测试初始模型的拟合度。

class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi   ##np.vstack 是 NumPy 库中的一个函数，用于垂直堆叠数组,所有输入数组必须有相同的列数（即第二维大小相同）
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B)  # residues:残差和
        #sl.lstsq 函数（代表“least squares”，即最小二乘法）用于解决线性最小二乘问题
        # 。这个函数可以找到一组参数，使得一个线性方程组的残差平方和最小化。这在统计学和机器学习中非常常见，尤其是在线性回归分析中。
        #x：解向量，即使得 | | Ax - B | | 最小化的向量。返回拟合参数x、残差平方和resids（每个数据点的预测值和实际值之间的差异）、矩阵的秩rank和奇异值s。
        return x  # 返回最小平方和向量 （参数向量 x 使得模型预测值和实际数据值之间的残差平方和最小化）

    def get_error(self, data, model): #它是 LinearLeastSquareModel 类的一部分。这个方法的目的是计算线性模型预测值和实际值之间的误差，
        # 具体来说，是计算每个数据点的残差平方和。 model（拟合出的模型参数）
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi   .T 是转置操作，将 A 矩阵转置，使得每个输入变量成为 A 的一行。
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = sp.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b  ###sp.dot(A, model) 是矩阵 A 和模型参数 model 的点积，相当于线性方程
        # y=Ax 的计算，其中 x 是模型参数。
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row  ##包含了每个数据点的残差平方和。
        # np.sum(..., axis=1) 是沿着矩阵的行（axis=1）计算平方残差的和
        return err_per_point  ##err_per_point：每个点的误差平方和。


def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    ##将这个数组乘以 20，得到一个形状为 (500, 1) 的数组，其中的元素是 0 到 20 之间的随机数
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    ##p.random.normal 是 NumPy 库中的一个函数，用于生成服从正态分布（也称为高斯分布）的随机数
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:  #if 1: 总是为真，所以下面的代码块总是会执行。
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        ##使用 np.arange 生成一个从 0 到 A_noisy 数组的行数减 1 的整数序列，这个序列包含了数据集中所有样本点的索引。
        #
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
     ##使用 np.hstack 将 A_noisy 和 B_noisy 水平堆叠起来，形成一个新数组 all_data。
     ##这个数组包含了所有的输入和输出数据，形状为 (500, 2)，即 500 行 2 列。
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False##设置 debug 模式为 False，这意味着在模型拟合过程中不会输出额外的调试信息
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型
    #实例化 LinearLeastSquareModel 类，创建一个线性模型对象 model；input_columns 和 output_columns 指定了输入和输出数据的列索引。
    # input_columns 和 output_columns 指定了输入和输出数据的列索引。

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    # linear_fit：线性模型的参数，即最小二乘解。
    # resids：残差平方和。
    # rank：输入矩阵的秩。
    # s：输入矩阵的奇异值。
    # run RANSAC 算法

    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)
##这是 n 参数，表示每次迭代中随机选择的样本点数量，用于拟合模型,1000=1000,这是 t 参数，表示阈值，用于确定一个数据点是否符合当前的模型
    #300     #d 参数，表示拟合较好时需要的最少内点数量。
    #当设置为 True 时，函数将返回两个值：最佳拟合模型和包含内点索引的数据。

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组
        ##这段代码的目的是对 A_exact 数组中的数据点按照 x 坐标进行排序。这在绘图时特别有用，因为它可以帮助我们按照 x 值的顺序绘制数据点，从而更清晰地展示数据的分布和趋势。

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图

            #绘制 A_noisy 和 B_noisy 数组的散点图，表示所有噪声数据点。'k.' 表示黑色点标记。
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")

            #仅绘制被 RANSAC 算法识别为内点的数据点，使用蓝色叉号标记。
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data') #绘制非局外点的数据点，使用黑色点标记。
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data') #绘制局外点的数据点，使用红色点标记。

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit') #绘制 RANSAC 算法拟合的直线，使用排序后的 x 坐标和 RANSAC 拟合参数。
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system') #绘制理想情况下的直线，使用排序后的 x 坐标和完美的拟合参数。
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit') #绘制线性回归拟合的直线，使用排序后的 x 坐标和线性回归拟合参数
        pylab.legend() #显示图例，包括每条线或数据点集的标签。
        pylab.show() #显示最终的绘图结果。


if __name__ == "__main__":
    test()



