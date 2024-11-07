import numpy as np
import pylab
import scipy.linalg as sl
import scipy as sp

class LinearLeastSquareModel:
    def __init__(self, input_column, output_column):
        self.input_column = input_column
        self.output_column = output_column

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_column]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_column]).T  # 第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T #第一列Xi-->行Xi
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T #第二列Yi-->行Yi
        B_fit = np.dot(A, model) #计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum( (B - B_fit) ** 2, axis = 1 ) #sum squared error per row
        return err_per_point

def createData():
    """
    制造理想的数据
    @return:
    """
    data_nums = 500  # 数据总量
    input_num = 1  # 输入变量个数
    output_num = 1  # 输出变量个数
    # 随机生成行X,坐标都是0-20
    X_loc = 20 * np.random.random((data_nums, input_num))
    # 生成随机的正太分布的线性参数
    k = 60 * np.random.normal(size=(input_num, output_num))
    # y = x*k
    Y_loc = np.dot(X_loc, k)
    # 横坐标，纵坐标都添加额外的点，最小二乘法也好处理的
    X_loc_noise = X_loc + np.random.normal(size=(X_loc.shape))
    Y_loc_noise = Y_loc + np.random.normal(size=(Y_loc.shape))
    # 上边基本都是内群的点，这里再添加点外群的点
    outer_num = 100  # 点的数量
    # 全部x的坐标
    all_ids = np.arange(X_loc_noise.shape[0])
    # 随机打乱
    np.random.shuffle(all_ids)
    outer_ids = all_ids[:outer_num]
    X_loc_noise[outer_ids] = 20 * np.random.random((outer_num, input_num))
    Y_loc_noise[outer_ids] = 50 * np.random.normal(size=(outer_num, output_num))
    # 这时候数据已经造好了，然后将数据变成坐标的形式
    alldata = np.hstack((X_loc_noise, Y_loc_noise))
    input_columns = range(input_num)  # 数组的第一列x:0
    output_columns = [input_num + i for i in range(output_num)]  # 数组最后一列y:1
    # 初始化模型
    model = LinearLeastSquareModel(input_columns, output_columns)
    pylab.plot(X_loc_noise[:, 0], Y_loc_noise[:, 0], 'k.', label='data')  # 散点图
    return alldata,model,X_loc_noise,Y_loc_noise


def ransac(data, model, n, k, t, d, return_all=False):
    """
    1.数据中随机选n个点设定为内群
    2.根据这几个点，得到合适的模型，我们这里是通过点得到线性回归方程 y=kx
    3.将其它没选择到的点带入到刚才建立的模型中，计算是否为内群，hi-y => ri  得到残差，设置阈值
    4.记下内群数量，
    5.重复以上步骤
    6.比较哪次计算中，内群数量最多，，内群最多的，就是我们需要的
    @return:
    """
    iterations = 0
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    # 循环次数大于等于k时，停止循环
    while iterations < k:
        all_ids = np.arange(data.shape[0])
        np.random.shuffle(all_ids)
        # #数据中随机选n个点设定为可能的内群，maybe_idxs为内群，其它的点待测试
        maybe_idxs = all_ids[:n]
        test_idxs = all_ids[n:]
        print("maybe_idxs\n",maybe_idxs)
        # 获取(maybe_idxs)行数据(Xi,Yi)
        maybe_inliers = data[maybe_idxs:]
        test_points = data[test_idxs]  # 其它群的（X,y）
        #最小二乘法拟合模型 2.根据这几个点，得到合适的模型，
        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points,maybemodel)
        # 小于误差的都算内群的点索引
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]#得到内群点
        if (len(also_inliers) > d):
            # 如果内群的点个数大于d，则将其拼接到内群
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            # 如果平均误差下雨最少误差，则更新群内的点
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
        iterations += 1
    #没有合适的模型，则返回错误
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit,{'inliers':best_inlier_idxs}
    else:
        return bestfit

def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data) #获取n_data下标索引
    np.random.shuffle(all_idxs) #打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

if __name__ == '__main__':
    alldata,model,A_noisy,B_noisy = createData()
    ransac_fit, ransac_data = ransac(alldata, model, 50, 1000, 7e3, 300,return_all=True)
    pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, 0], all_data[:, 1])
