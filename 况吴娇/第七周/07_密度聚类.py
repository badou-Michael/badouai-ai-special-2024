import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度

##：这个符号表示选择所有的行，即不改变行的选择，也就是选择所有的150个样本
# :4：这个符号表示选择从第一列到第四列（包括第四列）的所有列。由于索引是从0开始的，所以 :4 实际上选择了第0列、第1列、第2列和第3列。
print(X.shape)
# 绘制数据分布图
'''
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

# 
# X[:, 0]：表示选择 X 数组中所有行（样本）的第一列（特征），即所有样本的第一个特征值，通常代表萼片长度（sepal length）。
# X[:, 1]：表示选择 X 数组中所有行（样本）的第二列（特征），即所有样本的第二个特征值，通常代表萼片宽度（sepal width）。
# c="red"：设置点的颜色为红色。
# marker='o'：设置点的形状为圆形。
# label='see'：为这些点设置图例标签，这里标签为 'see'。
'''

'''
切片与单个索引的区别：
切片（Slicing）：使用 X = iris.data[:, 0:4] 这种形式，其中 0:4 表示一个范围，
从索引0开始到索引4结束（不包括索引4）。这是一个左闭右开的区间，意味着包括起始索引0，但不包括结束索引4。
单个索引（Indexing）：使用 X = iris.data[:, 4] 这种形式，直接指定一个索引值4，表示选择第五列
'''

dbscan= DBSCAN(eps=0.4,min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_

print('label_pred',label_pred)


'''
DBSCAN(eps=0.4, min_samples=9) 参数解释：
eps：这是 DBSCAN 算法中的一个重要参数，表示邻域的大小。在这个例子中，eps=0.4 意味着算法会考虑距离每个点 0.4 以内的所有点作为该点的邻居。
min_samples：这个参数表示形成一个聚类所需的最小样本数。在这个例子中，min_samples=9 意味着如果一个点的邻居数量达到或超过 9 个，那么这个点就会被认为属于一个聚类的核心点。
dbscan.fit(X)：
这行代码是使用 DBSCAN 算法对数据集 X 进行拟合（训练）。fit 方法会根据 DBSCAN 的规则来对数据点进行聚类。
label_pred = dbscan.labels_：
dbscan.labels_ 是 DBSCAN 聚类结果的一个属性，它是一个数组，包含了每个数据点的聚类标签。每个标签都是一个整数，表示该数据点属于哪个聚类。如果一个点被标记为噪声（即不属于任何聚类），则其标签为 -1。
'''

'''
loc 参数用于指定图例的位置。matplotlib 定义了一些标准的缩写，用于快速设置图例的位置。这些缩写基于图表的四个象限：

loc='best'（默认）：自动选择最不碍事的位置。
loc='upper right'（loc=1）：将图例放置在图表的右上角。
loc='upper left'（loc=2）：将图例放置在图表的左上角。
loc='lower left'（loc=3）：将图例放置在图表的左下角。
loc='lower right'（loc=4）：将图例放置在图表的右下角。
loc='right'：将图例放置在图表的右侧。
'''
# 获取所有独特的标签
unique_labels = np.unique(label_pred)
print(f'unique_labels: {unique_labels}')
# #确定聚类的数量：
# 计算聚类的数量（排除噪声标签 -1）
num_clusters = len(unique_labels[unique_labels > -1])
print(f"Number of clusters: {num_clusters}")


x0=X[label_pred==0]
x1=X[label_pred==1]
x2=X[label_pred==2]

plt.scatter(x0[:,0],x0[:,1],c='red',marker='o',label='label0') #这里，x0[:, 0] 和 x0[:, 1] 分别表示聚类 0 中所有数据点的第一个和第二个特征值，即横坐标和纵坐标。
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length') #：设置 x 轴的标签为 "sepal length"（萼片长度）。
plt.ylabel('sepal width') #设置 y 轴的标签为 "sepal width"（萼片宽度）。
plt.legend(loc=2)  #plt.legend
plt.show()


'''
x0 = X[label_pred == 0]、x1 = X[label_pred == 1]、x2 = X[label_pred == 2]：
这些行代码是根据聚类标签来筛选数据点。label_pred == 0 创建了一个布尔数组，其中属于聚类 0 的位置为 True，其他为 False。
然后，X[...] 根据这个布尔数组选择相应的行，即属于聚类 0 的所有数据点。类似地，x1 和 x2 分别选择了属于聚类 1 和聚类 2 的数据点。
'''
'''
plt.scatter 参数解释：
x0[:, 0]、x0[:, 1]：这些是散点图的横坐标和纵坐标数据，分别表示属于聚类 0 的数据点的第一个和第二个特征值。
c="red"、c="green"、c="blue"：这些参数设置散点图中点的颜色。
marker='o'、marker='*'、marker='+'：这些参数设置点的形状。
label='label0'、label='label1'、label='label2'：这些参数为不同聚类的点设置图例标签。
plt.xlabel、plt.ylabel：
这些函数设置 x 轴和 y 轴的标签。
'''

'''
plt.scatter 是 matplotlib.pyplot 模块中的一个函数，用于绘制散点图。散点图是一种用于展示两个变量之间关系的图表，通过在二维平面上绘制点来表示数据点的分布。

函数原型：
python
plt.scatter(x, y, s=None, c=None, marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors=None, **kwargs)
参数解释：
x：横坐标数据，可以是一个数值列表或数组。
y：纵坐标数据，可以是一个数值列表或数组。
s：点的大小，可以是单个数字或与 x 和 y 长度相同的数组。如果为 None，则使用默认大小。
c：点的颜色，可以是单个颜色值、颜色列表或数组。如果为 None，则使用默认颜色。
marker：点的形状，如 'o'（圆形）、's'（正方形）、'^'（三角形）等。
cmap：颜色映射对象，用于将数值映射到颜色。
norm：归一化对象，用于调整颜色数据的范围。
vmin：颜色数据的最小值。
vmax：颜色数据的最大值。
alpha：点的透明度。
linewidths：点的边框宽度。
edgecolors：点的边框颜色。
kwargs：其他关键字参数，用于自定义点的样式。
'''