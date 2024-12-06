
import numpy as np

list_1 = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]


def normalization_1(list_input):
    '''归一化（0~1）'''
    '''x=(x−x_min)/(x_max−x_min)'''
    return [(float(num) - min(list_input)) / (float(max(list_input)) - min(list_input)) for num in list_input]


def normalization_2(list_input):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    return [(float(num) - np.mean(list_input)) / (float(max(list_input)) - min(list_input)) for num in list_input]


def z_score(list_input):
    '''零均值归一化: 经过处理后的数据均值为0，标准差为1（正态分布）'''
    ''' :y=(x−μ)/σ 其中μ是样本的均值， σ是样本的标准差'''
    x_mean = np.mean(list_input)
    s2 = sum([(num - np.mean(list_input) ) **2 for num in list_input]) / len(list_input)
    return [(num - x_mean) / np.sqrt(s2) for num in list_input]


test_list = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
print(test_list)

n1 = normalization_1(test_list)
print(n1)

n2 = normalization_2(test_list)
print(n2)

n3 = z_score(test_list)
print(n3)
