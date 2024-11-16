###cluster.py
#导入相应的包
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
import matplotlib.pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''
# 定义数据集 X，每个元素是一个二维坐标点
X = [[1,2],[3,2],[4,4],[1,2],[1,3],[5,6],[8,3]]
# linkage函数用于计算层次聚类的连接矩阵。
# fcluster函数用于根据给定的阈值从层次聚类的树状结构中提取聚类。
# dendrogram函数用于绘制层次聚类的树状图。
Z = linkage(X,"ward")  #“average”（平均连接法）、“complete”（最远距离法）和 “single”（最近距离法）“ward”（最小方差法）
f = fcluster(Z, t=8, criterion='distance')  #fcluster函数接受连接矩阵Z、距离阈值t和判断准则criterion（'distance'表示根据距离判断，或者'inconsistent'等）作为参数
fig = plt.figure(figsize=(8,5))  # 创建一个图形，尺寸为 (8, 5)
dn = dendrogram(Z)
print(Z)
plt.title('Dendrogram')
plt.xlabel('Data points')
plt.ylabel('Distance')
plt.show()
