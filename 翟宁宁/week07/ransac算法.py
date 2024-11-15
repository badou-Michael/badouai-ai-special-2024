
'''
1. 生成测试数据集
2. 创建模型  Y = kX,
3. 执行模型

'''

import numpy as np
import scipy as sp
import scipy.linalg as sl

import pylab
'''
    1.生成测试数据集
'''
def start_test():
    '''
    生成500个数据集，400个是正常点，其余为较大的噪声或无效点
    '''

    data_size = 500 #500 样本数据集逼近某条直线y= kx+b ,目标求出参数k,b .使得估算直线逼近真实方程
    x_inputs = 1
    y_outputs = 1

    x_data = 20* np.random.random(size = (data_size,x_inputs))  #随机生成0-60之间的500个数据:行向量 （500*1） Xi
    true_fit = 60* np.random.random(size = (x_inputs,y_outputs))  #真实的斜率
    y_data = np.dot(x_data,true_fit)   # (500*1)行向量 Yi        y=kx


    #添加高斯噪声
    x_noisy_data = x_data + np.random.normal(size=x_data.shape) #Xi
    y_noisy_data = y_data + np.random.normal(size=y_data.shape) #Yi

    '''
    加入局外点all_out_point = 100 ,我们拿到数据集的数据序号500个 ，把顺序打乱 ，取前100个，把这点定为局外点
    '''
    all_out_point_number = 100   #局外点个数
    all_point = np.arange(x_noisy_data.shape[0]) # 取出所以序号0-499
    np.random.shuffle(all_point)  #打乱序号
    all_out_point = all_point[:all_out_point_number]   #100局外点初始化

    x_noisy_data[all_out_point] = 20*np.random.normal(0,1,size=(all_out_point_number,x_inputs))
    y_noisy_data[all_out_point] = 50*np.random.normal(0,1,size=(all_out_point_number,y_outputs))

    '''
        设置模型
    '''
    all_point_data = np.hstack( (x_noisy_data, y_noisy_data) ) #形式([Xi,Yi]....) shape:(500,2) 500行2列
    input_columns = range(x_inputs)  #数组的第一列x:0
    output_columns = [x_inputs + i for i in range(y_outputs)] #数组最后一列y:1

    model = LinearLeastModel(input_columns, output_columns)


    #最小二乘法拟合
    linear_fit,resids,rank,s = sl.lstsq(all_point_data[:,input_columns],all_point_data[:,output_columns])




    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_point_data, model, 50, 1000, 7e3, 300,  return_all=True)




    # 画图

    sort_idxs = np.argsort(x_data[:, 0])
    A_col0_sorted = x_data[sort_idxs]  # 秩为2的数组

    if 1:
        pylab.plot(x_noisy_data[:, 0], y_noisy_data[:, 0], 'k.', label='data')  # 散点图
        pylab.plot(x_noisy_data[ransac_data['inliers'], 0], y_noisy_data[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    else:
        pylab.plot(x_noisy_data[non_outlier_idxs, 0], y_noisy_data[non_outlier_idxs, 0], 'k.', label='noisy data')
        pylab.plot(x_noisy_data[outlier_idxs, 0], y_noisy_data[outlier_idxs, 0], 'r.', label='outlier data')
    pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,ransac_fit)[:,0],
                    label='RANSAC fit' )
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, true_fit)[:, 0],
               label='exact system')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, linear_fit)[:, 0],
               label='linear fit')
    pylab.legend()
    pylab.show()


'''
2. 创建模型  Y = kX,
    这里在做什么？  第一点我们知道真实数据是符合线性的规律，这是已知到的，问题是我们如何用数学中线性模型描述这个规律 。
    就是使用一个Function来描述这个规律,我们要拟合的函数是y=kx  ,k  = ?  多少呢 ，
    我们来调节这个k的值，使得拟合函数与真实函数无限接近（残差和） ，那就可以认为这个k值是最好的 
'''
class  LinearLeastModel:
    #利用最小二乘法求线性AX=B的解  ，该模型作为ransac模型输入
    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns



    def fit(self, data):
        '''
        data数据样本[[X1,Y1],[X2,Y2],........]
        vstack把它变成列向量 ，Xi,Yi


        np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        residues:残差和
        k 返回最小平方和向量
        '''
		#
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T #第一列Xi-->行Xi
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T #第二列Yi-->行Yi
        k, resids, rank, s = sl.lstsq(A, B) #residues:残差和
        return   k



    '''
    k值已只，计算残差和  （真实数据与拟合数据）
    '''
    def get_error(self, data, model):
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T  #第一列Xi-->行Xi
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T  #第二列Yi-->行Yi
        B_fit = sp.dot(A, model)  #计算的y值, B_fit = A * model.k
        err_per_point = np.sum( (B - B_fit) ** 2, axis = 1 ) #残差和
        return err_per_point



''''
3. 执行模型

'''
def ransac(data, model, n, k, t, d, debug = False, return_all = False):
    """
       输入:
           data - 样本点
           model - 假设模型:事先自己确定
           n - 生成模型所需的最少样本点
           k - 最大迭代次数
           t - 阈值:作为判断点满足模型的条件
           d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
       输出:
           bestfit - 最优拟合解（返回nil,如果未找到）

    """

    iterations = 0   # 迭代次数
    bestfit = None    #最优拟合解（返回nil,如果未找到）
    besterr = np.inf #设置默认值  正无穷大
    best_in_idxs = None



    '''
    k次迭代找的最优拟合解 
    1. 在数据中随机选择几（n）个点设定为内群
    '''
    while iterations<k:

        #1. 在数据中随机选择几（n）个点设定为内群
        all_indx = np.arange(data.shape[0])
        np.random.shuffle(all_indx)
        maybe_index = all_indx[:n]     #内群索引
        test_index = all_indx[n:]   #外群索引


        #2. 计算适合内群的模型
        maybe_in_points = data[maybe_index,:]
        test_points = data[test_index,:]
        print("test_index:",test_index)
        #拟合模型
        k = model.fit(maybe_in_points)


        #把非内群点带入，残差和，如果某一个点小于给定的阈值，则更新该点为内群
        err_num = model.get_error(test_points, k)  # 残差和
        print("err_num:",err_num<t)
        also_indx = test_index[err_num<t]
        also_points = data[also_indx,:]

        if len(also_points) >d :
            better_data = np.concatenate((maybe_in_points,also_points))  # 合并最好的数据
            better_k =model.fit(better_data)
            better_err = model.get_error(better_data, better_k)

            this_err = np.mean(better_err)

            if this_err < besterr:
                besterr = this_err
                bestfit = better_k
                best_in_idxs  = np.concatenate((also_indx,maybe_index))


        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit,{'inliers':best_in_idxs}
    else:
        return bestfit



if __name__ == "__main__":
    start_test()
