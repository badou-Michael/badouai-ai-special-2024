### cluster_modified.py
# 导入所需的库
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

# linkage函数用于执行层次聚类，它有三个参数：
# 1. y：距离矩阵，可以是一维压缩向量（距离向量）或二维观测向量（坐标矩阵）。
#    如果y是一维压缩向量，则y必须包含n个初始观测值的组合，n是坐标矩阵中成对观测值的数量。
# 2. method：计算类间距离的方法。
# 3. metric：度量标准，用于计算观测值之间的距离。

# fcluster函数用于根据给定的阈值或条件从层次聚类结果中提取扁平聚类：
# 1. Z：linkage函数得到的矩阵，记录了层次聚类的层次信息。
# 2. t：聚类的阈值，用于形成扁平聚类。
# 3. criterion：聚类形成标准，例如'inconsistent'。

# 定义数据集X，每个子列表代表一个观测值的坐标。
X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]

# 使用ward方法计算类间距离，并对数据集X进行层次聚类。
Z = linkage(X, 'ward')

# 根据距离阈值4来形成聚类，这里使用'distance'作为聚类形成标准。
f = fcluster(Z, 4, 'distance')

# 创建一个图形对象，并设置图形大小。
fig = plt.figure(figsize=(5, 3))

# 绘制树状图，展示层次聚类的结果。
dn = dendrogram(Z)

# 打印linkage矩阵Z，查看层次聚类的详细信息。
print(Z)

# 显示图形。
plt.show()
