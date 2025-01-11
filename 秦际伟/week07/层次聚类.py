# 导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

# 观测数据点
data_points = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]

# 进行层次聚类
linkage_matrix = linkage(data_points, method='ward')

# 生成平面聚类结果
cluster_labels = fcluster(linkage_matrix, 4, criterion='distance')

# 绘制树状图
fig = plt.figure(figsize=(5, 3))
dendrogram(linkage_matrix)

# 添加标题和标签
plt.title('层次聚类树状图')
plt.xlabel('数据点索引')
plt.ylabel('距离')

# 打印聚类层次信息矩阵
print("聚类层次信息矩阵:\n", linkage_matrix)

# 显示树状图
plt.show()

# 保存图像
plt.savefig('dendrogram.png')
