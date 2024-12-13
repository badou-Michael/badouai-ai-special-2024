import numpy as np
import math

def Normalization1(x):
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def Normalization2(x):
    return [(float(i) - np.mean(x)) / float(max(x) - min(x)) for i in x]  # 分母上的float是有必要的吗？

# 这个是方差吧
def z_score(x):
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x))*(i - np.mean(x)) for i in x ]) / len(x)
    # print(s2)
    return [(i - x_mean) / s2 for i in x]

# 标准差
def z_score_2(x):
    x_mean = np.mean(x)
    s2 = sum([((i - np.mean(x)) ** 2) for i in x]) / len(x)
    # print(s2)
    s = math.sqrt(s2)
    # print(s)

    # s2 = np.var(data)
    # print(s2)
    # s = np.std(x)
    # print(s)

    return [(i - x_mean) / s for i in x]

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

data_after = Normalization1(data)
print(data_after)

data_after_2 = Normalization2(data)
print(data_after_2)

data_after_3 = z_score(data)
print(data_after_3)

data_after_4 = z_score_2(data)
print(data_after_4)
