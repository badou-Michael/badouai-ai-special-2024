#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：Ransac.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/11/07 12:13
'''
import numpy
import scipy as sp
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt




'Step 1: 用来生成数据 random_partition 函数'
def random_partition (n, n_data):
    '''
    这个函数的作用就是把咱们给定的数据随机分成两部分呀，一部分用来拟合模型(就是随机选择的点来拟合模型)，另一部分用来测试模型(拟合模型之外的点，用来带入拟合好的模型后筛选出内群点和离群点)
    @param n: 表示要从总的数据里随机选取出来，用于去拟合模型的那部分数据的数量。
    （比如说有 100 个数据点，设置 n 为 10 的话，意味着每次在 RANSAC 算法的迭代过程中，会先随机挑出 10 个数据点来尝试拟合出一个可能的模型）

    @param n_data: 这个参数代表的是总的数据的数量（比如有100个数据点，就传入100）
    （它的作用就是让函数知道数据的总量有多少，这样才能准确地根据 n 的值去划分数据，
    先把所有的索引（从 0 到 n_data - 1）都获取到，再从中选取前 n 个索引对应的那部分数据用于拟合模型，剩下的用于测试模型）

    @return: index1 index2
    '''
#先用 np.arange(n_data) 获取了从 0 到 n_data - 1 的所有索引，然后用 np.random.shuffle 把这些索引打乱顺序
    # 此段打乱索引的作用在于：后续每一次循环中取n个数计算模型，都在原始数据中得到和上一轮循环不同的n个随机数，（也就是循环一次，随机再取一遍n个数）
    allData_index = np.arange(n_data)
    np.random.shuffle(allData_index)

    #把前 n 个索引作为一组（用来拟合模型的那部分数据的索引），剩下的作为另一组（用来测试模型的那部分数据的索引）返回
    index1 = allData_index[:n]
    index2 = allData_index[n:]
    return index1,index2

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

'Step 2: 定义计算模型的方法 LinearLeastSquareModel 类 (最小二乘法)'
class LinearLeastSquareModel:
    '''
    这个类呢是用来定义线性模型相关操作，有怎么拟合模型、怎么计算误差这些功能
    '''

    def __init__(self, input_columns, output_columns, debug = False):
        '''
        input_columns 和 output_columns 这两个参数分别用来指定在处理数据时，哪些列是作为输入特征（也就是自变量相关的数据所在列），哪些列是作为输出（也就是因变量相关的数据所在列）
        @param input_columns: 作为x的簇的索引 ，存放的是索引值，是个一维数组，比如input_columns指定为[0，1]表示原始数据当中的第一列和第二列的数据作为自变量x
        @param output_columns: 作为y的簇的索引
        '''
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        '''
        这个 fit 方法的功能就是根据给定的数据来拟合出一个线性模型
        @param data:
        @return: x
        '''
        # 先获取到 x 和 y 这两列的数据
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T

        # 然后使用最小二乘法来拟合出一个线性模型
        # sl.lstsq(A, B) 这个函数用来解线性方程 ，
        x, resids, rank, s = sl.lstsq(A, B)
        return x

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = np.dot(A,model)
        err_per_point = np.sum( (B - B_fit) ** 2, axis = 1 )

        return err_per_point

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ransac 核心代码
def ransac (data, model, n, k, t, d, debug = False, return_all = False):
    '''

    @param data: 待分析的数据
    @param model: 通过某种方法拟合的模型,事先要定义好这个模型
    @param n: 生成模型所需的最少样本点
    @param k: 最大迭代次数
    @param t: 作为 （除开已选n个点之外的点，代入生成模型后，判断是否满足模型） 的误差阈值
    @param d: 用于筛选出内点数量大于设定阈值d的模型
    @param debug:
    @param return_all:
    @return: bestfit - 最优拟合解（返回nil,如果未找到）
    '''
    # 初始化一些参数
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    # 开始迭代 RANSAC 算法
    while iterations < k:
        # 随机从数据中获取 n 个数据点，以及除了n之外的
        maybe_idxs, text_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :] # 可能的内点 ，也就是随机取的n，用来计算拟合模型
        test_points = data[text_idxs] # 测试点，用来代入拟合模型，然后筛选出内外点

        # 尝试用最小二乘法来拟合出一个线性模型
        maybe_model = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybe_model) # 得到test_points所有点与拟合模型的maybe_model的残差平方和

        # also_idxs就是除了传入的50个原始计算点（n点）之外的并且小于阈值t的test_points点，也记录为内群点
        also_idxs = text_idxs[test_err < t]
        also_inliers = data[also_idxs, :]

        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
            # if len(also_inliers > d):
        print('d = ', d)

        if len(also_inliers > d): # 筛选出内点数量大于设定阈值d的模型,
            betterdata = np.concatenate ((maybe_inliers,also_inliers))
            bettermodel = model.fit(betterdata)
            bettererr = model.get_error(data, bettermodel)
            thiserr = np.mean(bettererr) # 取当前模型的残差平方和的平均数（也就是平均误差）作为衡量最优模型的标准
            if thiserr < besterr: #本次循环的误差与上一次循环的误差作比较 besterr一开始默认设置为无穷大（besterr = np.inf）
                bestfit = bettermodel
                besterr = thiserr # 更新besterr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs)) # 返回本次循环记录内点的索引标号
        iterations += 1 # 一直更新迭代数，（直到等于传参K值后停止）

        if bestfit is None:  # 表示循环结束后bestfit依旧没有更新，代表一条“直线”（模型）都没有找到，说明可能和设置的阈值有关，有可能阈值设置太高了（肯能是 len(also_inliers) > d的d；也可能是 test_idxs[test_err < t]的t；也可能是循环次数K值）
            raise ValueError("did't meet fit acceptance criteria")
        if return_all:  # 如果找到了最优解，则返回bestfit  return_all= True代表返回每一次循环计算出的模型，也就是所有模型
            return bestfit, {'inliers': best_inlier_idxs}
        else:
            return bestfit

        return bestfit


# TODO: 开始测试
def test():

    #1 生成一些模拟的数据，初始化以下变量
    n_data = 500 # 样本个数
    # 输入变量和输出变量的个数分别设置为1个（假设拟合的目标函数是y=k*x+b  就是一个自变量一个因变量）
    n_inputs = 1
    n_outputs = 1

    A_exact = 20 * np.random.random((n_data, n_inputs))  # 随机生成0-20之间的500个数据:行向量（也就是（500，1）的数组）
    print('0-20之间的500个随机数：A_exact',A_exact)

    # 生成一个大小为 (n_inputs, n_outputs) 即 (1, 1) 的数组，元素服从正态分布，这里可以理解为随机生成的理想直线的斜率，
    perfect_fit = 60 * np.random.normal( size = (n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    print('随机线性度：perfect_fit',perfect_fit)

    #利用矩阵乘法，根据理想的线性关系 y = kx（这里 k 就是 perfect_fit 里的值，x 是 A_exact 里的值 ）生成对应的因变量 y 值，得到理想状态下无噪声干扰的数据对
    B_exact = sp.dot(A_exact, perfect_fit)  # y = x * k

    # 2 加入高斯噪声
    #给理想的自变量数据 A_exact 加上服从正态分布的随机噪声，噪声的维度和 A_exact 一致，模拟真实世界中数据采集时混入的噪声，生成了带噪声的自变量数据
    A_noisy = A_exact + np.random.normal( size = A_exact.shape ) #500 * 1行向量,代表Xi
    #同样地，给理想的因变量数据 B_exact 加上服从正态分布的随机噪声，生成带噪声的因变量数据 ，此时的数据更贴近实际场景
    B_noisy = B_exact + np.random.normal( size = B_exact.shape ) #500 * 1行向量,代表Yi
    print('A_noisy:',A_noisy)
    print('B_noisy:',B_noisy)


    # 3 添加"局外点"
    if 1:
        n_outliers = 100 # 设定离群点数量是100个
        all_idxs = np.arange(A_noisy.shape[0])# 获取带噪声的所有数据索引0-499
        np.random.shuffle(all_idxs) # 把这些索引打乱顺序，方便后续随机抽取索引来选择数据点，后续就可以利用这些随机抽取的索引，将选中的数据点修改为局外点

        outlier_idxs = all_idxs[:n_outliers]  # 在被打乱的0-500个的点中选取前 n_outliers 个作为噪声点，这里取到的只是索引

        # 对于这些局外点索引对应的位置，重新随机生成自变量 A_noisy[outlier_idxs] 和因变量 B_noisy[outlier_idxs] 的值，使它们成为偏离原始线性关系的局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi

    #将带噪声和局外点的数据按列拼接成 all_data，形状为 (500, 2)
    all_data = np.hstack((A_noisy, B_noisy))

    # 定义输入列 input_columns 和输出列 output_columns，这里输入是第一列，输出是第二列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1


    debug = False

    # 实例化 LinearLeastSquareModel 类得到 model，用于后续 RANSAC 算法中的模型拟合与误差计算，当做传参传入ransac 函数中
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型

    # 利用 sp.linalg.lstsq 进行普通最小二乘拟合，得到 linear_fit 及其它相关信息，后续用于对比。
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    '以上已准备好ransac 的所有参数'
    '''
    调用 ransac 函数，
    传入数据 all_data、模型 model，以及参数 50（每次选的最少样本点数）、1000（最大迭代次数）、7e3（误差阈值）、300（最少内点数量） ，
    并设置返回所有信息。得到 ransac_fit（RANSAC 拟合结果）和 ransac_data（包含内点信息）
    '''
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)


    # 导入pylab 绘图
    '''
    对理想数据按自变量排序，得到 A_col0_sorted
    绘制多个图形：
    用黑色点绘制原始带噪声的数据 pylab.plot(A_noisy[:,0], B_noisy[:,0], 'k.', label = 'data')。
    用蓝色叉号绘制 RANSAC 算法筛选出的内点数据 pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label = "RANSAC data")。
    分别绘制 RANSAC 拟合直线、理想直线、普通最小二乘拟合直线，用于对比不同拟合方法的效果，最后添加图例并显示图形。通过对比，可以直观看到 RANSAC 算法在存在噪声与局外点情况下，能否比普通最小二乘更好地拟合数据
    '''
    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],np.dot(A_col0_sorted, perfect_fit)[:, 0],label='exact system')
        pylab.plot(A_col0_sorted[:, 0],np.dot(A_col0_sorted, linear_fit)[:, 0],label='linear fit')
        pylab.legend()
        pylab.show()

if __name__ == "__main__":
    test()