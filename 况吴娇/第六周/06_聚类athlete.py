# coding=utf-8
from sklearn.cluster import KMeans
##脚本首先导入了KMeans类，这是进行K均值聚类分析的核心类。KMeans是sklearn.cluster模块中的一个类，用于将数据点分组成K个簇。

##K-Means：这是最常用的聚类算法之一，旨在将数据分成 K 个簇，使得簇内距离（通常是欧几里得距离）最小化。
"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
"""
X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
     ]

# 输出数据集
print(X)
"""
第二部分：KMeans聚类
clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
"""
clf = KMeans(n_clusters=3)
y_pred=clf.fit_predict(X)

#输出完整Kmeans函数，包括很多省略参数
print(clf)
#输出聚类预测结果
print("y_pred = ",y_pred)


"""
第三部分：可视化绘图
"""

import numpy as np
import matplotlib.pyplot as plt

#获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in X]
print (x)
y = [n[1] for n in X]
print (y)

''' 
绘制散点图 
参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
'''
#plt.scatter(x, y, c=y_pred, marker='x')：这个函数用于在二维平面上绘制散点图

plt.scatter(x,y,c=y_pred,marker='*') ##c=y_pred 指定了每个点的颜色。y_pred 是一个数组，
# 包含了每个数据点的聚类预测结果，通常是一个整数标签，表示该点属于哪个类别。matplotlib 会根据这些标签自动为每个类别分配不同的颜色。

# 绘制标题
plt.title("Kmeans-Basketball Data")

# 绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

# 设置右上角图例
plt.legend(["A","B","C"]) #["A","B","C"]：图例中的标签 列表。在这个例子中，图例包含三个标签，分别对应于散点图中的三个不同类别或聚类预测结果。

##plt.legend 函数不能写成 plt.legend("A", "B", "C")。这是因为 plt.legend 函数期望接收一个标签列表作为其参数，
# 而不是多个单独的字符串参数。这个列表可以是字符串列表，也可以是 matplotlib.lines.Line2D 对象的列表，或者是两者的组合。
#显示图形
plt.show() #plt.show()：这个函数用于显示图形。在调用这个函数之前，所有的绘图命令都是设置图形的状态，但不会立即显示图形。调用 plt.show() 后，会弹出一个窗口显示绘制的图形。

##这段代码的目的是绘制一个篮球数据的 K-Means 聚类结果，其中 x 轴表示每分钟助攻数（assists_per_minute），
# y 轴表示每分钟得分（points_per_minute），散点图中的点用叉号表示，并且根据聚类预测结果 y_pred 用不同的颜色区分不同的类别。图例显示在图形的右上角，用于标识不同的类别。






