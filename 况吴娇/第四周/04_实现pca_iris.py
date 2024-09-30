#!/usr/bin/env python
# encoding=gbk

##import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import  load_iris


x ,y = load_iris(return_X_y=True) #加载数据，x表示数据集中的属性数据，y表示数据标签
pca=dp.PCA(n_components=2) #加载pca算法，设置降维后主成分数目为2  #pca = sklearn.decomposition.PCA(n_components=2)
reduced_x=pca.fit_transform(x)  #降维后的数据 #fit_transform方法的下划线确实表示了方法执行的先后顺序。，保存在reduced_x中

#1初始化空列表：
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
#2,遍历降维后数据：
for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    #二维数组，其中每行代表一个数据点，每列代表一个特征。pca.fit_transform(x) 将这个原始数据集通过 PCA 降维，结果仍然是一个二维数组，其中每行代表一个降维后的数据点，每列代表一个主成分。
    #reduced_x的长度（即len(reduced_x)的值）等于原始数据集中的数据点数量。

    # 在数据集中，特征和标签通常以以下形式组织：
    # 特征矩阵：一个二维数组，其中每一行代表一个样本，每一列代表一个特征。
    # 标签数组：一个一维数组，其中的每个元素对应特征矩阵中每一行样本的标签。标签数组:['Setosa', 'Sonata', 'Versicolor']   /[0, 1, 2] 0 代表 "Setosa"，1 代表 "Sonata"，2 代表 "Versicolor"
#1初始化空列表：这些列表用于存储不同种类鸢尾花的降维后坐标。
    # reduced_x[i]：：表示第 i 个降维后的数据点（一个包含多个特征的数组或列表）。
    # reduced_x[i][0]：表示第i个数据点的第一个特征。
    # reduced_x[i][1]：表示第i个数据点的第二个特征。
    if y[i] == 0:  #根据标签（y[i]）进行判断，因为这代表了每个数据点的“正确答案”或类别。在鸢尾花数据集的例子中，标签告诉我们每个样本属于哪个种类的鸢尾花。
        red_x.append(reduced_x[i][0])##在绘图时使用 append 方法，是因为我们通常需要将数据点的坐标收集到列表中，以便后续可以一次性地将它们绘制到图表上。
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:####尽管我们通过PCA降维到2个主成分，数据集本身包含3个类别的鸢尾花
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()


 # sklearn.decomposition是scikit-learn（简称sklearn）库中的一个模块，它提供了一系列用于数据降维和特征提取的方法。
# # PCA (Principal Component Analysis): 主成分分析，用于通过线性变换将数据投影到新的坐标系中，以捕捉数据中最大的方差。

# 参数 return_X_y 是 Scikit-learn 中数据加载函数的一个参数，用于指定函数返回数据的方式。当设置为 True 时，函数返回两个分开的数组：X（特征数据）和 y（目标数据或标签）。
# 参数名中的大写字母 X 和小写 y 是为了区分它们所代表的不同部分。X 通常用来表示特征数据集，而 y 表示目标数据集或标签数组。