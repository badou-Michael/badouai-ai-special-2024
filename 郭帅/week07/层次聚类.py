#层次聚类

from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

#设置数据
x = [[1,2],[3,2],[6,4],[3,6],[5,0],[9,9],[6,7],[2,5],[9,9]]
#进行层次聚类
z = linkage(x,'ward')
#给处理簇的结果,两种方式：distance 按距离阈值，maxclust 按簇的数量
f = fcluster(z,2,'distance')
#创建画框
fig = plt.figure(figsize=(5, 3))
#在上面画框绘制由 z 生成的树状图
dn = dendrogram(z)
#显示图画
plt.show()
