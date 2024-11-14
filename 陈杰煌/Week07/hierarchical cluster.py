from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

'''
采用最小距离的凝聚层次聚类算法流程：

(1) 将每个对象看作一类，计算两两之间的最小距离；
(2) 将距离最小的两个类合并成一个新类；
(3) 重新计算新类与所有类之间的距离；
(4) 重复(2)、(3)，直到所有类最后合并成一类为止。
'''

'''
linkage(y, method='single', metric='euclidean') 包含以下3个参数:

1. y: 距离矩阵, 可以是一维的压缩向量(距离向量), 也可以是二维的观测向量(坐标矩阵). 
        如果 y 是一维压缩向量, 则它必须是 n 个初始观测值的组合, 其中 n 是坐标矩阵中成对的观测值数量.
2. method: 用于计算类间距离的方法.
3. metric: 用于计算距离的度量方式.
'''

'''
fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)

1. 第一个参数 Z 是由 linkage 得到的矩阵，记录了层次聚类的层次信息。
2. t 是一个聚类的阈值，用于形成扁平聚类时应用（"The threshold to apply when forming flat clusters"）。
'''

# 使用层次聚类中的 linkage 函数生成聚类树（凝聚树）
# 参数：
# X - 原始数据坐标矩阵
# 'ward' - 使用 Ward 最小方差方法计算距离

data_points = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
linkage_matrix = linkage(data_points, method='ward')

# 使用 fcluster 根据给定的阈值（距离）生成扁平聚类
# linkage_matrix - linkage 输出的聚类信息矩阵
# 4 - 距离阈值，设置成 4，用于决定多少个簇
# 'distance' - 以距离为准将样本分配给不同簇
clusters = fcluster(linkage_matrix, t=4, criterion='distance')

# 生成并绘制树状图，展示聚类的层次结构
fig = plt.figure(figsize=(5, 3))
dendrogram(linkage_matrix)
plt.show()

# 输出聚类信息矩阵
print(linkage_matrix)