# -*- coding: utf-8 -*-
# time: 2024/10/18 16:28
# file: PCA.py
# author: flame
# 导入matplotlib.pyplot模块，用于绘制图形
import matplotlib.pyplot as plt

# 导入sklearn.decomposition模块中的PCA类，用于执行主成分分析
import sklearn.decomposition as dp

# 从sklearn.datasets._base模块中导入load_iris函数，用于加载鸢尾花数据集
from sklearn.datasets._base import load_iris

# 调用load_iris函数加载鸢尾花数据集，返回特征数据x和标签y
x, y = load_iris(return_X_y=True)

# 创建PCA对象，设置n_components=2，表示将数据降维到2维
pca = dp.PCA(n_components=2)

# 使用PCA对象的fit_transform方法对特征数据x进行主成分分析，并将结果存储在reduced_x中
# fit_transform方法会先拟合数据，然后进行转换
reduced_x = pca.fit_transform(x)

# 初始化三个列表，分别用于存储三类数据点的x坐标和y坐标
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

# 遍历主成分分析后的数据
for i in range(len(reduced_x)):
    # 根据标签y的值，将数据点分配到不同的列表中
    if y[i] == 0:
        # 如果标签为0，将数据点的x坐标和y坐标分别添加到red_x和red_y列表中
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        # 如果标签为1，将数据点的x坐标和y坐标分别添加到blue_x和blue_y列表中
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        # 如果标签为2，将数据点的x坐标和y坐标分别添加到green_x和green_y列表中
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

# 使用plt.scatter方法绘制红色散点图，表示第一类数据点
# 参数c='r'表示颜色为红色，marker='x'表示标记为叉号
plt.scatter(red_x, red_y, c='r', marker='x')

# 使用plt.scatter方法绘制蓝色散点图，表示第二类数据点
# 参数c='b'表示颜色为蓝色，marker='D'表示标记为菱形
plt.scatter(blue_x, blue_y, c='b', marker='D')

# 使用plt.scatter方法绘制绿色散点图，表示第三类数据点
# 参数c='g'表示颜色为绿色，marker='.'表示标记为点
plt.scatter(green_x, green_y, c='g', marker='.')

# 使用plt.show方法显示绘制的散点图
plt.show()
