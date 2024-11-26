import numpy as np
import scipy as sp
import scipy.linalg as sl
 
def ransac(data, model, n, k, t, d, debug = False, return_all = False):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点[内点]最少的个数,当做阈值看待
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
    iterations = 0 # 迭代次数，初始化为零
    bestfit = None # 最优拟合解，初始化为空
    besterr = np.inf # 最优误差/最小误差，设置默认值，将误差变量 besterr 初始化为正无穷大
    best_inlier_idxs = None # 最佳的内点索引初始化为空
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0]) 
        # 将数据集随机划分为两个子集（n是第一个子集的大小，data.shape[0]是数据集的总大小）
        # 将数据集 data 随机划分为两个子集：可能的内点索引集/训练集的索引 maybe_idxs 和测试索引集/测试集的索引 test_idxs；
        print ('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :] #获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs] #若干行(Xi,Yi)数据点，和上一句本质一样，取出所有索引对于的行，其中列为所有列
        maybemodel = model.fit(maybe_inliers) #拟合模型，计算出经过这些点的直线                       
        test_err = model.get_error(test_points, maybemodel) #计算误差:平方和最小
        print('test_err = ', test_err < t) # test_err = [[ True],[False],[ True],[False]]
        also_idxs = test_idxs[test_err < t] # 从 test_idxs 中提取那些误差小于阈值 t 的点的索引，存储在 also_idxs 中
        print ('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs,:] # 从数据集中提取那些误差小于阈值 t 的内点，存储在 also_inliers 中
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',numpy.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)) ) # 打印当前迭代次数和内点数量。%d 是格式化字符串，用于插入整数
        # if len(also_inliers > d):
        print('d = ', d)
        if (len(also_inliers) > d):
            betterdata = np.concatenate( (maybe_inliers, also_inliers) ) #样本连接，即该模型maybemodel较好，得到该模型下的所有内点
            bettermodel = model.fit(betterdata) #拟合模型，再次拟合得到新模型
            better_errs = model.get_error(betterdata, bettermodel) # 再次计算新模型的误差
            thiserr = np.mean(better_errs) #平均误差作为新的误差
            if thiserr < besterr: # 本次误差比上一次的误差/初始误差 小
                bestfit = bettermodel # 更新最佳模型
                besterr = thiserr # 更新最佳误差
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) ) #更新局内点,将新点加入
        iterations += 1 # 迭代完一次，迭代次数加一，总共迭代k次
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria") # 如果最终 最优模型为空，则报错
    if return_all: # 根据 布尔值return_all 来决定返回的内容
        return bestfit,{'inliers':best_inlier_idxs} 
        #返回一个[包含两个元素的]元组，第一个元素是 bestfit，第二个元素是一个字典，键为 'inliers'，值为 best_inlier_idxs
    else:
        return bestfit # 返回最优模型
 
 
def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data) #获取n_data下标索引，创建一个包含从0到 n_data-1 的整数数组
    np.random.shuffle(all_idxs) #打乱下标索引，随机打乱 all_idxs 数组中的元素顺序
    idxs1 = all_idxs[:n] #取出前 n 个索引，【左闭右开】，若n=4，则取出的是[0,1,2,3]
    idxs2 = all_idxs[n:] #取出第 n 个索引到最后一个索引，【左闭右开】
    return idxs1, idxs2
 
class LinearLeastSquareModel:
    #最小二乘求线性解,用于RANSAC的输入模型    
    def __init__(self, input_columns, output_columns, debug = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    
    def fit(self, data):
        # data: 输入的数据矩阵，假设是一个二维的 NumPy 数组，其中每一行代表一个样本，每一列代表一个特征。
		# np.vstack 按垂直方向（行顺序(在此代码中是原本的列)）堆叠数组 并转置构成一个新的数组 111
        # np.vstack 之所以被称为“垂直”堆叠，是因为它在二维数组中沿着行的方向（即 y 轴方向）进行堆叠，并转置。这种命名方式有助于直观地理解数组的堆叠方式。
        '''
        a = np.array([1, 2])
        b = np.array([3, 4])
        result = np.vstack((a, b))
        得到
        array([[1, 2],
               [3, 4]])
        '''
        '''
        假设 data 是一个 5x5 的矩阵，而 input_columns=[1, 3]：
        data = np.array([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]
        ])
        
        列表推导式：
        data[:, 1]=array([ 1,  6, 11, 16, 21])
        data[:, 3]=array([ 3,  8, 13, 18, 23])
        即[array([ 1,  6, 11, 16, 21]), array([ 3,  8, 13, 18, 23])]

        np.vstack 会将这两个数组按行顺序堆叠，并转置（.T），得到：
        array([
            [ 1,  3],
            [ 6,  8],
            [11, 13],
            [16, 18],
            [21, 23]
        ])
        '''
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T #第一列Xi-->行Xi，input_columns=第1列
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T #第二列Yi-->行Yi，output_columns=第2列
        x, resids, rank, s = sl.lstsq(A, B) 
        # 最小二乘法来求解线性方程 A * x = B，其中 A 是输入矩阵，B 是输出矩阵
        # x 是求解得到的系数矩阵，表示输入特征与输出特征之间的线性关系；residues:残差和；rank 是矩阵 A 的秩；s 是奇异值；
        return x #返回最小平方和向量   
 
    def get_error(self, data, model):
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T #第一列Xi-->行Xi
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T #第二列Yi-->行Yi
        B_fit = sp.dot(A, model) #计算的y值,B_fit = model.k*A + model.b，将输入矩阵 A 与模型参数矩阵 model 进行矩阵乘法运算，得到结果B_fit
        err_per_point = np.sum( (B - B_fit) ** 2, axis = 1 ) #sum squared error per row，按行求和[前面已经用np.vstack()从每1列转每1行,故最后得到[行数]个数字]，得到每个数据点的误差平方和z
        return err_per_point
 
def test():
    '''
    1,生成理想数据：
    1-1,准备好真实数据 B_exact = sp.dot(A_exact, perfect_fit)，即y = x * k
    1-2,准备好训练集 A_noisy，B_noisy，先经过加高斯噪声，再随机将100个点替换为局外点
    '''
    n_samples = 500 #样本个数
    n_inputs = 1 #输入变量个数
    n_outputs = 1 #输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))#随机生成0-20之间的500个数据:行向量：生成一个形状为 (500, 1) 的数组，每个数为 [0, 1) 区间内的随机浮点数，然后乘以 20
    perfect_fit = 60 * np.random.normal( size = (n_inputs, n_outputs) ) #随机线性度，即随机生成一个斜率：
    # 生成一个包含单个元素的二维数组（即一个形状为 (1, 1) 的数组），该元素是一个服从均值为 0、标准差为 60 的正态分布的随机浮点数。
    # np.random.normal 是 NumPy 提供的一个函数，用于生成服从标准正态分布（均值为 0，标准差为 1）的随机浮点数

    B_exact = sp.dot(A_exact, perfect_fit) # y = x * k
 
    #加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal( size = A_exact.shape ) #500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal( size = B_exact.shape ) #500 * 1行向量,代表Yi
 
    #添加"局外点"
    if 1:
        n_outliers = 100
        all_idxs = np.arange( A_noisy.shape[0] ) #获取索引0-499
        np.random.shuffle(all_idxs) #将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers] #100个0-500的随机局外点：打乱索引后，取出前100个索引，即随机取出100个索引作为局外点的索引
        A_noisy[outlier_idxs] = 20 * np.random.random( (n_outliers, n_inputs) ) #加入噪声和局外点的Xi：
        # 使用 outlier_idxs 作为索引，将 A_noisy 数组中对应位置的元素替换为新的随机值，这些随机值的范围是 [0, 20*1)
        B_noisy[outlier_idxs] = 50 * np.random.normal( size = (n_outliers, n_outputs)) #加入噪声和局外点的Yi
    
    '''
    2,开始训练：
    2-1,准备好模型，用LinearLeastSquareModel
    2-2,
    '''
    #setup model 
    all_data = np.hstack( (A_noisy, B_noisy) ) #形式([Xi,Yi]....) shape:(500,2) 即500行2列
    # np.hstack(): 用于水平方向上的数组拼接，即在列方向上增加新的列。
    # np.vstack(): 用于垂直方向上的数组拼接，即在行方向上增加新的行。
    input_columns = range(n_inputs)  #数组的第一列x:0，
    output_columns = [n_inputs + i for i in range(n_outputs)] #数组最后一列y:1
    #（引申，n_inputs=1，n_outputs=1，则input_columns=[0]，output_columns=[1]）
    #（引申，n_inputs=5，n_outputs=5，则input_columns=[0, 1, 2, 3, 4]，output_columns=[5, 6, 7, 8, 9]）
    #（引申，n_inputs=5，n_outputs=2，则input_columns=[0, 1, 2, 3, 4]，output_columns=[5, 6]）
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug = debug) # 自定义的类的实例化:用最小二乘生成已知模型
 
    linear_fit,resids,rank,s = sp.linalg.lstsq(all_data[:,input_columns], all_data[:,output_columns]) 
    # 用来拟合一条直线到一组数据点。解决线性最小二乘问题
        # linear_fit: 这是拟合得到的线性模型的系数。对于简单的线性回归，linear_fit 将包含两个元素，分别是斜率和截距。！！！
        # resids: 这是残差，即实际数据点与拟合直线之间的差异。
        # rank: 这是矩阵的秩，表示输入数据的线性独立性。
        # s: 这是奇异值，表示输入数据矩阵的条件数。如果条件数很大，说明输入数据可能存在多重共线性问题。
    
    #run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug = debug, return_all = True) # 自定义的类
    # ransac(data, model, n, k, t, d, debug = False, return_all = False)
    # 输入:
        # data 样本点，model 假设模型:事先自己确定，n 生成模型所需的最少样本点，k 最大迭代次数，
        # t 阈值:作为判断点满足模型的条件，d 拟合较好时,需要的样本点[内点]最少的个数,当做阈值看待
    # 输出:
        # bestfit - 最优拟合解（返回nil,如果未找到），best_inlier_idxs - 最优解的内点索引
 
    if 1:
        import pylab
 
        sort_idxs = np.argsort(A_exact[:,0]) # np.argsort 函数返回的是排序后的索引
        A_col0_sorted = A_exact[sort_idxs] # 秩为2的数组？秩为1吧：A_col0_sorted 是一个与 A_exact 形状相同的数组，但行的顺序是根据 A_exact[:,0] 的值从小到大升序排列的
 
        if 1:
            pylab.plot( A_noisy[:,0], B_noisy[:,0], 'k.', label = 'data' ) # 所有训练点的散点图：'k.'黑点
            pylab.plot( A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label = "RANSAC data" ) # 其中所有内点的散点图： 'bx'蓝色叉号
        else:
            pylab.plot( A_noisy[non_outlier_idxs,0], B_noisy[non_outlier_idxs,0], 'k.', label='noisy data' )
            pylab.plot( A_noisy[outlier_idxs,0], B_noisy[outlier_idxs,0], 'r.', label='outlier data' )
 
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,ransac_fit)[:,0],
                    label='RANSAC fit' )  # 为什么要将A_exact进行排序：为了绘制出的直线是有序的，并且perfect_fit 只有斜率 截距=0
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,perfect_fit)[:,0],
                    label='exact system' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,linear_fit)[:,0],
                    label='linear fit' )
        pylab.legend()
        pylab.show()
 
if __name__ == "__main__":
    test()
