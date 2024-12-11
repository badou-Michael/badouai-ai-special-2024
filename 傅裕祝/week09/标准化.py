import numpy as np

# 归一化方法一
def normalize(data):
    num = []
    for i in data:
        num.append((float(i) - np.mean(data)) / float(max(data) - min(data)))
    return num

# 归一化方法二
def normalize2(data):
    return [(float(i) - min(data)) / float(max(data) - min(data)) for i in data]


# 标准化方法
def standardization(data):
    mean = np.mean(data)  # 求均值 μ
    std = np.std(data)  # 求标准差 σ
    print('标准差：', std)
    # 公式：z=(x-μ)/σ
    z = [(float(z) - mean) / std for z in data]
    return z


data = [10, 20, 30, 40, 50]
num1 = normalize(data)
num2 = normalize2(data)
standard_num = standardization(data)
print(num1)
print(num2)
print('标准化后数据:%s' % standard_num)
