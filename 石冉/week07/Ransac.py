import numpy as np

#定义ransac函数
def ransac(data,model,n,k,t,d,return_all=False):
    '''data代表输入的数据
    model代表输入数据服从的模型，这里使用最小二乘法，调接口
    n代表初始选择的内点个数
    k代表迭代次数
    t代表多大误差内属于内点
    d代表至少多少个内点才能拟合模型
    未设置debug
    '''
    iterations=0 #默认初始迭代次数为0
    bestfit=None #设置初始最佳拟合为空
    besterr=np.inf #设置初始最佳误差为无穷大
    best_inliner_idxs=None #设置初始内点为空

    #在小于k的次数内开始迭代
    while iterations<k:
        n_data=data.shape[0] #获取data第一行
        all_idxs=np.arange(n_data)  # 获取n_data下标索引
        np.random.shuffle(all_idxs)  # 打乱下标索引
        maybe_idxs=all_idxs[:n] #获取前n个数据集作为初始内点
        test_idxs=all_idxs[n:] #剩下的数据集作为测试数据集
        maybe_inliners=data[maybe_idxs,:]
        test_points=data[test_idxs]
        maybe_model=model.fit(maybe_inliners)#调用下方定义的最小二乘法拟合结果
        test_err=model.get_error(test_points,maybe_model)#计算误差
        also_idxs=test_idxs[test_err<t] #获取内点索引
        also_inliners=data[also_idxs,:] #获取内点

        #接下来比较获取的内点数是否超过d，如果超过d，将其赋值到bestfit、besterr
        if (len(also_inliners)>d):
            betterdata=np.concatenate((maybe_inliners,also_inliners)) #将初始随机选取的内点与拟合的内点拼接，再次拟合
            bettermodel=model.fit(betterdata)
            better_err=model.get_error(betterdata,bettermodel)
            this_err=np.mean(better_err)#取平均误差
            if this_err<besterr:
                bestfit=bettermodel
                besterr=this_err
                best_inliner_idxs=np.concatenate((maybe_idxs,also_idxs)) #更新内点
        iterations += 1
    if bestfit == None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all == True:
        return bestfit,{'inliners':best_inliner_idxs}
    else:
        return bestfit


#定义最小二乘法作为模型，取得拟合结果及误差
class LinearLeastSquareModel:
    def __init__(self,input_columns,output_columns):
        self.input_columns=input_columns
        self.output_columns=output_columns

    def fit(self,data):
        A=np.vstack([data[:,i] for i in input_columns]).T
        B=np.vstack([data[:,i] for i in output_columns]).T
        X,_,_,_=np.linalg.lstsq(A,B,rcond=None)
        return X

    def get_error(self,data,model):
        A = np.vstack([data[:, i] for i in input_columns]).T
        B = np.vstack([data[:, i] for i in output_columns]).T
        B_fit=np.dot(A,model)
        err_per_point=np.sum((B-B_fit)**2,axis=1)
        return err_per_point

# 使用示例
if __name__ == "__main__":
    # 生成理想数据
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs)) #随机生成500个20以内的数
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs)) #随机生成一个斜率
    B_exact = np.dot(A_exact, perfect_fit) #计算y

    # 加入高斯噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    # 添加局外点
    n_outliers = 100
    all_idxs = np.arange(A_noisy.shape[0])
    np.random.shuffle(all_idxs)
    outlier_idxs = all_idxs[:n_outliers]
    A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
    B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    # 设置模型
    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = [0]
    output_columns = [1]
    model = LinearLeastSquareModel(input_columns, output_columns)

    # 运行 RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, return_all=True)

    # 绘制结果
    import pylab

    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]

    pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
    pylab.plot(A_noisy[ransac_data['inliners'], 0], B_noisy[ransac_data['inliners'], 0], 'bx', label="RANSAC data")
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='exact system')
    pylab.legend()
    pylab.show()
