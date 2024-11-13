class LinearLeastSquareModel:
    # 最小二乘求线性解，用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠构成一个新的数组
        row = np.vstack([data[:, i] for i in self.input_columns]).T
        column = np.vstack([data[:, i] for i in self.output_columns]).T
        # x:最小二乘解，resids：残差平方和，rank: 输入矩阵的秩，s:输入矩阵的奇异值
        x, resids, rank, s = sl.lstsq(row, column) 
        return x
    def get_error(self, data, model_fit):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 根据model拟合出的model_fit值，计算B_fit = A * model_fit，从而求error
        B_fit = np.dot(A, model_fit) 
        err_per_point = np.sum((B - B_fit) ** 2, axis = 1)
        return err_per_point

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    params: 
            - data: 样本点
            - model: 假设模型
            - n : 生成模型所需的最少样本点
            - k : 最大迭代数
            - t : 阈值，作为判断点满足模型的条件
            - d : 拟合较好时，需要的样本点最少的个数，当作阈值看待
    return: 
            - bestfit : 最优拟合解，返回nil，如果未找到
    """
    iterations = 0
    bestfit = None
    besterr = np.inf # 设置默认值
    best_inlier_idxs = None
    while iterations < k:
        # 生成内群和其他的点
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs,: ]
        test_points = data[test_idxs]
        # 拟合模型
        maybeModel = model.fit(maybe_inliers)
        # 求误差
        test_err = model.get_error(test_points, maybeModel)
        # 求在误差内的点的下标
        also_idxs = test_idxs[test_err<t]
        also_inliers = data[also_idxs, :]
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',numpy.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)) )
        # 如果内群数量大于阈值
        if (len(also_inliers) > d):
            # 样本连接
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            # 连接后重新计算误差
            betterModel = model.fit(betterdata)
            betterError = model.get_error(betterdata, betterModel)
            thiserr = np.mean(betterError)
            # 如果有更优的结果，则更新
            if thiserr < besterr:
                bestfit = betterModel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit
        


def random_partition(n, n_data):
    """返回n个随机数据和剩下n_data - n个数据"""
    all_idxs = np.arange(n_data) # 获取n_data个下标
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

def helper():
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perfect_fit = 60 * np.random.normal(size = (n_inputs, n_outputs))# 随机生成一个斜率
    B_exact = np.dot(A_exact, perfect_fit) # y = x * k

    # 加入高斯噪声
    A_noisy = A_exact + np.random.normal(size = A_exact.shape)
    B_noisy = B_exact + np.random.normal(size = B_exact.shape)

    # 添加局外点
    if 1:
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0]) # 获取0-499索引
        np.random.shuffle(all_idxs) # 将all_indxs打乱
        outliers_idxs = all_idxs[:n_outliers] # 随机选择100个局外点
        A_noisy[outliers_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        B_noisy[outliers_idxs] = 40 * np.random.normal( size = (n_outliers, n_outputs))
    
    all_data = np.hstack( (A_noisy, B_noisy) )
    debug = False
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]
    model = LinearLeastSquareModel(input_columns, output_columns, debug = debug)

    # 调用sl.lstsq求最小二乘线性拟合
    linear_fit,resids,rank,s = sl.lstsq(all_data[:,input_columns], all_data[:,output_columns])

    # 调用RANSAC算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug = debug, return_all = True)

    if 1:
        import pylab
        
        sort_idxs = np.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_idxs] #秩为2的数组
        
        if 1:
            pylab.plot( A_noisy[:,0], B_noisy[:,0], 'k.', label = 'data' ) #散点图
            pylab.plot( A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label = "RANSAC data" )
        else:
            pylab.plot( A_noisy[non_outlier_idxs,0], B_noisy[non_outlier_idxs,0], 'k.', label='noisy data' )
            pylab.plot( A_noisy[outlier_idxs,0], B_noisy[outlier_idxs,0], 'r.', label='outlier data' )
        
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,ransac_fit)[:,0],
                    label='RANSAC fit' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,perfect_fit)[:,0],
                    label='exact system' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,linear_fit)[:,0],
                    label='linear fit' )
        pylab.legend()
        pylab.show()
if __name__ == '__main__':
    helper()