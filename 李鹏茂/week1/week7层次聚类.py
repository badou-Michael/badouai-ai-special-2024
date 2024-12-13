import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# 随机生成5个点，坐标范围在1到5之间
np.random.seed(0)  # 为了可重复性
points = np.random.uniform(1, 5, (5, 2))  # 5个二维点，坐标范围1到5

print("随机生成的5个点坐标：")
print(points)

# 计算距离矩阵（使用欧氏距离）
# 在层次聚类中，使用linkage函数时可以直接传入原始数据矩阵
Z = sch.linkage(points, method='ward')  # 'ward'是常用的聚类方法

# 可视化层次聚类的树形图
plt.figure(figsize=(8, 6))
sch.dendrogram(Z)
plt.title('层次聚类树形图')
plt.xlabel('样本')
plt.ylabel('距离')
plt.show()

# 通过树形图选择聚类的数量（例如，我们选择2个聚类）
clusters = sch.fcluster(Z, t=2, criterion='maxclust')

print("\n聚类结果（每个点对应的聚类标签）：")
for i, cluster in enumerate(clusters):
    print(f"点 {points[i]} 属于聚类 {cluster}")
