from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

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
"""
第二部分：KMeans聚类
clf = KMeans(n_clusters=3) 表示类簇数是3，聚成3类数据，clf赋值为KMeans
y_pred = clf.fit_predict(x) 载入数据集x，并且将聚类的结果赋值给y_pred
"""
clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)
"""
第三部分：数据可视化
"""
# 获取数据集的第一列和第二列数据 使用for循环获取n[0]表示x第一列
x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)
"""
绘制散点图
参数：x横轴，y纵轴，c=y_pred聚类预测结果
marker类型：o表示原点，* 表示星型 x表示点；
在plt.scatter函数中，c=y_pred告诉matplotlib根据y_pred数组中的类簇编号为每个点分配颜色。
默认情况下，matplotlib会使用一个颜色循环（color cycle）来为不同的类簇编号分配颜色。
matplotlib会使用默认的颜色映射（通常是viridis、plasma、inferno、magma、cividis之一）。
这个颜色映射定义了如何将类簇编号映射到具体的颜色上。每个类簇编号都会被映射到颜色映射中的一个颜色上。
matplotlib根据y_pred中的类簇编号和颜色映射，为每个点分配颜色，并绘制散点图。
"""
plt.scatter(x,y,c=y_pred,marker='x')
# 设置标签
plt.title("Kmeans-Basketball Data")
# 绘制x轴和y轴坐标
plt.xlabel('assists_per_minute')
plt.ylabel('points_per_minute')
# 设置右上角图例
plt.legend(['A','B','C'])
# 显示图像
plt.show()









