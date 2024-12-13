#-*- coding:utf-8 -*-
# author: 王博然
import numpy as np
import matplotlib.pyplot as plt

def normalization1(x): # 归一化 0～1
    # x_nm = (x-x_min)/(x_max-x_min)
    return [(float(i) - min(x))/float(max(x)-min(x)) for i in x]

def normalization2(x): # 归一化 -1～1
    # x_nm = (x-x_mean)/(x_max-x_min)
    x_mean = np.mean(x)
    return [(float(i) - x_mean)/float(max(x)-min(x)) for i in x]

def z_score(x):
    # x_nm = (x-μ)/σ
    x_mean = np.mean(x)
    s2 = sum([(float(i) - x_mean)**2 for i in x])/len(x)
    return [(i - x_mean)/s2 for i in x]

def draw(data, label):
    data_simplify = []
    cnt = []
    for i in data:
        if i not in data_simplify:
            data_simplify.append(i)
            c = data.count(i)
            cnt.append(c)
    # print(data_simplify)
    # print(cnt)
    plt.plot(data_simplify,cnt,label=label)
    

if __name__ == '__main__':
    data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    
    data_nm1 = normalization1(data)
    data_nm2 = normalization2(data)
    data_nm3 = z_score(data)

    # draw(data)
    draw(data_nm1, '0~1')
    draw(data_nm2, '-1~1')
    draw(data_nm3, 'zero-mean')
    plt.legend()
    plt.show()