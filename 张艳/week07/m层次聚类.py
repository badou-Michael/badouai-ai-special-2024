from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''

X = [[1,2],[3,2],[4,4],[1,2],[1,3]] # 横轴-每个点的序号：0->[1,2],1->[3,2],2->[4,4],3->[1,2],4->[1,3]；纵轴-两个点之间的距离
Z = linkage(X, 'ward') # 层次聚类结果 
f = fcluster(Z,4,'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()


'''
层次聚类结果:
第一列：被合并的第一个簇的索引（或编号）。
第二列：被合并的第二个簇的索引（或编号）。
第三列：两个簇之间的距离（或合并的代价）。
第四列：新生成的簇中包含的数据点数量。
[[0.         3.         0.         2.        ]
 [4.         5.         1.15470054 3.        ]
 [1.         2.         2.23606798 2.        ]
 [6.         7.         4.00832467 5.        ]]
'''
