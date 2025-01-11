#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/11/6 15:33
# @Author: Gift
# @File  : ransac.py
# @IDE   : PyCharm
import numpy as np
import matplotlib.pyplot as plt
import random


ITERS = 2000            # 最大迭代次数
SIZE = 50               # 样本数量
RATIO = 0.6             # 期望为内点的比例
INLIERS = SIZE * RATIO  # 内点

# 生成样本数据
X = np.linspace(0, 5, SIZE)
print(X)
Y = 2 * X + 5 #预设y=2x+5
Y1 = 2 * X + 5
print(Y)
for index in range(SIZE): #添加随机噪声
    sigma = np.random.uniform(-0.5, 0.5)  # 生成高斯噪声
    Y1[index]= Y[index] + sigma
print(Y1)

# 绘散点图
plt.figure()
plt.scatter(X, Y1)
plt.xlabel("x")
plt.ylabel("y1")

# 使用 RANSAC 算法估算模型
iter = 0  # 迭代次数
max_inliers = 0  # 最多内点数量
best_a = 0  # 最优参数
best_b = 0
error = 0.25  # 允许最小误差

while iter <= ITERS and max_inliers < INLIERS:

    # 随机选取两个点，计算模型参数
    random_index = random.sample(range(0, SIZE), 2)  # 返回索引列表
    x1 = X[random_index[0]]
    y1 = Y1[random_index[0]]
    x2 = X[random_index[1]]
    y2 = Y1[random_index[1]]

    a = (y2 - y1) / (x2 - x1)  # 斜率
    b = y1 - a * x1  # 截距
    inliers = 0  # 本次内点数量

    # 代入模型，计算内点数量
    for index in range(SIZE):
        y_estimate = a * X[index] + b
        if abs(Y1[index] - y_estimate) <= error:
            inliers += 1

    if inliers >= max_inliers:
        best_a = a
        best_b = b
        max_inliers = inliers

    iter += 1


# 画出拟合直线
print(best_a)
print(best_b)
Y2 = best_a * X + best_b
#plt.plot(X, Y1, linewidth=2.0, color="r")
plt.scatter(X, Y1, color="r",label="noise data")
plt.scatter(X, Y, color="g",label="true data")
#plt.plot(X, Y, 'o', color="b") #原始正确参数
plt.plot(X,Y,color="g",label="true line")
plt.plot(X,Y2,color="r",label="ransac line")
text = "best_a: " + str(round(best_a, 2)) + "\nbest_b:  " + str(round(best_b, 2)) + \
       "\nmax_inliers: " + str(int(max_inliers)) + "\nfinal_inter:" + str(iter)
plt.text(3, 6, text, fontdict={'size': 10, 'color': 'r'})
plt.title("RANSAC")
plt.legend()
plt.show()
