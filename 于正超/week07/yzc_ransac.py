"""
yzc-ransac
1.输入观测数据
2.选择随机点设置为内群-->计算合适内群的模型
    -->其他点带入上一步计算的模型中-->记录符合模型的内群数量
    -->重复以上步骤，计算哪次计算内群数量最多，即为最优解
"""
import numpy as np
import scipy as sp
import scipy.linalg as sl

#ransac
def ransac(data,model,n,k,t,d,debug=False,return_all = False):
    iterations = 0 #迭代次数
    bestfit = None  #最优拟合解
    besterr = np.inf  #设置默认值，返回inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_indexs , test_indexs = random_partition(n,data.shape[0])
        print('test_indexs = ',test_indexs)
        maybe_inliers = data[maybe_indexs,:]
        test_points = data[test_indexs]
        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points,maybemodel) ##计算误差，平方和最小
        print('test_err = ',test_err < t)
        also_indexs = test_indexs[test_err < t]
        print('also_indexs = ',also_indexs)
        also_inliers = data[also_indexs,:]
        if debug:
            print('test_err.min():',test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        print('d = ',d)
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers,also_inliers)) #连接随机点+验证出来的点
            bettmodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata,bettmodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettmodel
                besterr = thiserr
                best_inlier_idxs  = np.concatenate((maybe_indexs,also_indexs)) ##更新局内点，将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError('ERROR: did\'t meet fit acceptance criteria')
    if return_all:
        return bestfit,{'inliers':best_inlier_idxs}
    else:
        return bestfit


def random_partition(n,n_data):
    all_indexs = np.arange(n_data)
    np.random.shuffle(all_indexs)  #打乱索引
    indx1 = all_indexs[:n]  #索引分两份，一部分用来拟合模型，一部分用来计算误差验证准确性
    indx2 = all_indexs[n:]
    return indx1,indx2

#最小二乘法函数模型
class LinearLeastSquareModel:
    def __init__(self,input_columns,output_columns,debug = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self,data):  #用于拟合模型，返回模型参数,即训练模型
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T
        x,resids,rank,s = sl.lstsq(A,B)  #residues:残差和  ### x为最小二乘解；resids为残差平方和
        return x
    def get_error(self,data,model):  #用于评估模型，返回每个样本的误差平方和
        A = np.vstack([data[:,i] for i in self.input_columns] ).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = np.dot(A, model)
        err_per_point = np.sum( (B - B_fit) ** 2 ,axis=1 )
        return err_per_point



def test():
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples,n_inputs))##随机生成0-20之间的100个数据，行向量
    perfect_fit = 60 * np.random.normal(size= (n_inputs,n_outputs))  #随机线性度，即随机生成一个斜率
    B_exact = np.dot(A_exact,perfect_fit)  #y = x * k
    # print("x: {} \nk: {} \ny:{}".format(A_exact,perfect_fit,B_exact))
    # print(A_exact.shape)
    #加高斯噪声，最小二乘法也能很好的处理（可不加）
    A_noisy = A_exact + np.random.normal(size = (n_samples,n_inputs))
    B_noisy = B_exact + np.random.normal(size = B_exact.shape)
    # print(A_noisy)
    # print("B_noisy------",B_noisy)
    #添加局外点（噪声、外群、离群数据）
    if 1:
        n_outliers = 3
        all_indexs = np.arange(A_noisy.shape[0]) #获取数据点的索引
        np.random.shuffle(all_indexs)  #将索引打乱，截取部分索引值插入局外点
        out_indexs = all_indexs[0:n_outliers]
        A_noisy[out_indexs] = 20 * np.random.random((n_outliers,n_inputs))  ##等同于for i in out_indexs
        B_noisy[out_indexs] = 50 * np.random.normal(size=(n_outliers,n_outputs))
        # print(A_noisy)
        # print("B_noisy------", B_noisy)
    all_data = np.hstack((A_noisy,B_noisy))
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for  i in range(n_outputs)]
    debug = False
    print(input_columns)
    print("output----->",output_columns)

    model = LinearLeastSquareModel(input_columns,output_columns,debug=debug)
    print(model)
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    ransac_fit,ransac_data = ransac(all_data,model,50,100,7e3,300,debug=debug,return_all=True)
    if 1:
        import pylab

        sort_indexs = np.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_indexs]

        if 1:
            pylab.plot(A_noisy[:,0],B_noisy[:,0],'k.',label='data')
            pylab.plot(A_noisy[ransac_data['inliers'],0],B_noisy[ransac_data['inliers'],0],'bx',label = "Ransac Data")

        pylab.plot(A_col0_sorted[:,0],
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
    test()
