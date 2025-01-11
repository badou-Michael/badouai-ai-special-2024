# 归一化
import numpy as np


def Normalization(x):
    # => [0, 1)
    return [(np.array(i) - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0)) for i in x]

def Normalization2(x):
    # => (-1, 1)
    return [2 * (np.array(i) - np.mean(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0)) for i in x]

def z_score(x):
    # => 均值 0 标准差 1 正态分布
    mean = np.sum(x, axis=0) / len(x)
    print(mean)
    x = [(i - mean) for i in x]
    print(x)
    t = [ i ** 2 for i in x]
    print(t)
    std_devs = (np.sum(t, axis=0) / len(x)) ** 0.5
    print(std_devs)
    print(mean)
    x = x / std_devs
    return x

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
# l=[[1, 2], [2, 4], [3, 6]]
z = Normalization(l)
print(z)
z = Normalization2(l)
print(z)
z=[1, 2]
l = z_score(l)
print(l)

img = np.zeros((28, 28))
for i in range(10, 16):
    img[ :,i] = 1
print(img)
img = img.reshape((1, 28 * 28))

imgs = 2 * [img]
print(img)
print(img.shape)