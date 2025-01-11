import numpy as np
import scipy as sp
from scipy.stats import truncnorm

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    Args:
        data    :样本点
        model   :假设模型（最小二乘）
        n       :生成模型需要的最小样本点
        k       :k轮迭代
        t       :判断点是否满足模型的条件
        d       :所需要的最少类内样本点
        debug (bool, optional): _description_. Defaults to False.
        return_all (bool, optional): _description_. Defaults to False.
    """
    # model要有两个函数：fit(data), get_error(data, model)
    m = 0
    best_model = None
    best_error = float("inf")
    while(m < k):
        test_idxs, maybe_idxs = random_partition(n, data)
        maybe_data = data[maybe_idxs, :]
        model_tmp1 = model.fit(maybe_data)
        test_data = data[test_idxs, :]
        # test_error是data，每个maybe_data计算一个error
        test_errors = model.get_error(test_data, model_tmp1)
        # 类内点
        also_data = test_data[test_errors < t]
        if len(also_data) > d:
            total_data = np.concatenate((test_data, also_data))
            model_tmp2 = model.fit(total_data)
            tmp_errors = model.get_error(total_data, model_tmp2)
            tmp_error = np.mean(tmp_errors)
            if tmp_error < best_error:
                best_error = tmp_error
                best_model = model_tmp2
        m += 1
    if best_model is None:
        raise ValueError("can't find model")
    else:
        print("model K: %f, d: %f"%(best_model[0], best_model[1]))
        return best_model
    
# 用来分割数据，一部分是拟合数据，一部分是测试数据
def random_partition(n, data):
    indics = np.arange(len(data))
    np.random.shuffle(indics)
    maybe_idxs = indics[:n]
    test_idxs = indics[n:]
    return test_idxs, maybe_idxs

class LeastSquareModel:
    def __init__(self):
        pass
    
    def fit(self, data):
        x = np.array(data[:,0])
        y = np.array(data[:,1])
        n = len(data)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        # sum_xy = np.sum(x*y)
        # sum_x2 = np.sum(x**2)
        # denominator = n*sum_x2-sum_x**2
        # assert denominator != 0, print("除零错误")
        k = sum_y / sum_x 
        # k = (n*sum_xy-sum_x*sum_y)/denominator
        # b = sum_y/n - k*sum_x/n
        return np.array([k,0.0])
    
    def get_error(self, data, model_par):
        errors = []
        for x,y_t in data:
            error = (y_t - (model_par[0]*x+model_par[1]))**2
            errors.append(error)
        return np.array(errors)
    
def test():
    n_samples = 500 # 样本个数
    n_inputs = 1 # 输入变量个数
    n_outputs = 1   # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # x的矩阵, 500行*1列
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # k
    B_exact = sp.dot(A_exact, perfect_fit)  # y = x*k
    
    A_noise = A_exact+np.random.normal(size=A_exact.shape)  # Xi
    B_noise = B_exact+np.random.normal(size=B_exact.shape)  # Yi
    
    B_min = np.min(B_noise, axis=0).item()
    B_max = np.max(B_noise, axis=0).item()
    
    mu = (B_min + B_max) / 2
    sigma = abs(B_max - B_min) / 4 + 1e-7
    # a和b是标准化的截断范围，标准的正态分布是均值为0，标准差为1
    a = (B_min - mu)/sigma if perfect_fit.item() > 0 else (B_min + mu)/sigma
    b = (B_max - mu)/sigma if perfect_fit.item() > 0 else (B_max - mu)/sigma
    
    if 1:
        n_outliers = 150
        # 将随机的100索引的值变成局外点
        all_index = np.arange(len(A_noise))
        np.random.shuffle(all_index)
        outlier_index = all_index[:n_outliers]
        A_noise[outlier_index] = 20*np.random.random((n_outliers, n_inputs))
        # B_noise[outlier_index] = np.random.normal(size=(n_outliers, n_outputs))
        B_noise[outlier_index] = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=(n_outliers, n_outputs))
    # hstack()用于将一系列数组按水平方向组合成一个新数组
    all_data = np.hstack((A_noise, B_noise))
    model = LeastSquareModel()
    input_columns = range(n_inputs) # 数据的前n_inputs个列是输入
    output_columns = [n_inputs+i for i in range(n_outputs)] # 数据的后面都是输出
    
    linear_fit, resids, rank, s = sp.linalg.lstsq(
        all_data[:, input_columns], all_data[:, output_columns])
    
    ransac_fit = ransac(all_data, model, 50, 1000, 7e3, 300)
    
    if 1:
        import pylab
        
        sort_idxs = np.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_idxs]
        if 1:
            pylab.plot(A_noise[:,0], B_noise[:,0], "k.", label="data")  #散点图
        print(ransac_fit)
        ransac_fit = ransac_fit[np.newaxis, :]
        A_design = np.column_stack((A_col0_sorted[:,0], np.ones_like(A_col0_sorted[:,0])))
        pylab.plot(A_col0_sorted[:,0],np.dot(A_design, ransac_fit.T)[:,0],label="RANSAC_fit")
        pylab.plot(A_col0_sorted[:,0],np.dot(A_col0_sorted, linear_fit)[:,0], label="linear_fit")
        pylab.plot(A_col0_sorted[:,0],np.dot(A_col0_sorted, perfect_fit)[:,0], label="perfect_fit")
        pylab.legend()
        pylab.show()
        
if __name__ == "__main__":
    test()