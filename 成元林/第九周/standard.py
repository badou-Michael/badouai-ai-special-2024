import numpy as np
import matplotlib.pyplot as plt
def normalization1(data):
    """
    最小值归一化，取值范围0-1
    公式：y = (x-min)/(max-min)
    @param data:
    @return:
    """
    return [(float(i)-min(data))/float(max(data)-min(data)) for i in data]

def normalization2(data):
    """
    均值归一化，取值范围-1-1
    公式：y = (x-mean)/(max-min)
    @param data:
    @return:
    """
    return [float(i-np.mean(data))/float(max(data)-min(data)) for i in data]

def zero_normalization(data):
    """
    zero-meannormalization 简称z-score
    零均值归一化
    经过处理后的数据均值为0，标准差为1（正态分布）
    公式：y = (x-μ)/σ
    其中μ是样本的均值，σ是样本的标准差
    @param data:
    @return:
    """
    mean = np.mean(data)
    s = np.sum([(i-mean)**2 for i in data])/len(data)
    return [(i-mean)/s for i in data]

if __name__ == '__main__':
    l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
         11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    l1 = []
    cs = []
    for i in l:
        c = l.count(i)
        cs.append(c)
    print(cs)
    n = normalization2(l)
    z = zero_normalization(l)
    print(n)
    print(z)
    '''
    蓝线为原始数据，橙线为z
    '''
    plt.plot(l, cs)
    plt.plot(z, cs)
    plt.show()

