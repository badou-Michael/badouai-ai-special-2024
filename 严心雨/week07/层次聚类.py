import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,fcluster,linkage

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。
   'single' 最短距离
'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]

#method='ward' 沃德方差最小化算法
#生成聚类树 层次聚类的层次信息
#矩阵Z第1、2列表示被两两连接生成一个新类的对象索引
#矩阵第3列表示两两对象的连接距离
#矩阵第4列表示当前类中原始对象的个数
Z = linkage(X,'ward')
#print(Z)

#聚类结果
#t=4 想分4类
f = fcluster(Z,4,'distance')

#画布大小
figure = plt.figure(figsize=(5,4))

#dendrogram() 聚类树图绘制
dst = dendrogram(Z)
plt.show()
