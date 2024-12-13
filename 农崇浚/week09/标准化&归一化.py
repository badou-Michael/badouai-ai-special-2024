import numpy as np


#Max_min归一化
def Min_Max(data):
    h, w = data.shape
    max = np.max(data)
    min = np.min(data)

    Min_max = np.zeros_like(data)
    for i in range(w):
        for j in range(h):
            Min_max[i,j] = (data[i,j] - min) / (max - min)

    return  Min_max

#标准化
def Norm(data):
    avg = np.sum(data)
    data_avg = np.zeros_like(data)
    h, w = data.shape
    for i in range(w):
        for j in range(h):
            data_avg[i,j] = (data[i,j] - avg) ** 2

    #方差
    s = np.sum(data_avg) / (h*w)
    #标准差
    s1 = s ** 0.5

    Norm_y = (data - avg) / s1

    return Norm_y
