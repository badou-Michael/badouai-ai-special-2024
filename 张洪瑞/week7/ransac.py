import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

# 最小二乘法求拟合函数
class Least_squares():
    def fit(self, data):
        Data_A = data[:, 0]
        Data_B = data[:, 1]
        N = Data_A.shape[0]
        k = (N * np.sum(np.dot(Data_A, Data_B)) - (np.sum(Data_A) * np.sum(Data_B))) / (N * np.sum(np.dot(Data_A, Data_A)) - np.sum(Data_A)**2)
        b = (np.sum(Data_B) / N) - (k * (np.sum(Data_A) / N))
        print("y = {}x+{}".format(k, b))
        return k, b

    def get_error(self, data):
        k, b = self.fit(data)
        Data_A = data[:, 0]
        Data_B = data[:, 1]
        Data_P = k * Data_A + b
        err = np.sum(np.dot((Data_P - Data_B), (Data_P - Data_B)))
        return err

# n:随机选几个点作为内群
# k:循环的轮次
# t:判断其他的点是否可以作为内群
# p:内群至少存在的样本点数量
def ransac(data, n, k, t, p):
    print(data)
    i = 0
    In_max = 0
    bestk = 0
    bestb = 0
    while i < k:
        Num_in = 0
        Num_p = n
        Data_A = []
        Data_B = []
        # 随机取n数量的点设为内群
        List = [i for i in range(500)]
        np.random.shuffle(List)
        List1 = List[:n]
        List2 = List[n:]
        for it in range(n):
            Data_A.append(data[List1[it]][0])
            Data_B.append(data[List1[it]][1])
        Data_A = np.array(Data_A)
        Data_B = np.array(Data_B)
        Data_A = Data_A.reshape((-1, 1))
        Data_B = Data_B.reshape((-1, 1))
        print(Data_A)
        Data_G = np.concatenate((Data_A, Data_B), axis=1)
        # 用最小二乘法取得可能的k和b的值
        maybek, maybeb = ls.fit(Data_G)
        # 取求得内群之外的其他点的误差
        for join in List2:
            # 如果误差在允许的范围内 则将此点加入到内群中
            if((maybek * data[join][0] + maybeb) - data[join][1] <= t):
                Num_in = Num_in + 1
                Num_p = Num_p + 1
        # 联合判断当内群点数量满足要求并且内群点的数量大于上一次迭代则更新参数
        if(Num_in > In_max and Num_p > p):
            In_max = Num_in
            bestk = maybek
            bestb = maybeb
    return bestk, bestb


if __name__ == '__main__':
    # 0-40随机数 random.random()(0-1)
    Data_x = 40 * np.random.random((500, 1))
    # 生成服从正态分布斜率numpy.random.normal(loc=0.0, scale=1.0, size=None)
    Slope = 10 * np.random.normal(size=(1, 1))
    # y = kx:Data_y = Slope * Data_x
    Data_y = np.dot(Data_x, Slope)
    # Data加噪声
    Noise_x = Data_x + np.random.normal(size=Data_x.shape)
    Noise_y = Data_y + np.random.normal(size=Data_y.shape)
    # 新增异常点
    List = [i for i in range(500)]
    np.random.shuffle(List)
    List = List[:100]
    for it in List:
        Noise_x[it] = 40 * np.random.random(size=(1, 1))
        Noise_y[it] = 50 * np.random.normal(size=(1, 1))

    Data = np.concatenate((Noise_x, Noise_y), axis=1)
    # 画线
    ls = Least_squares()
    k, b = ls.fit(Data)
    x = np.arange(0, 40, 0.1)
    y = k * x + b
    # 采用ransac方法
    bestk, bestb = ransac(Data, 50, 1000, 7e3, 300)
    x2 = np.arange(0, 40, 0.1)
    y2 = bestk * x2 + bestb

    plt.scatter(Noise_x, Noise_y)
    plt.plot(x, y, color='red', linewidth=2)
    plt.plot(x, y, color='green', linewidth=2)
    plt.xlabel("x-label")
    plt.ylabel("y-label")
    plt.show()
