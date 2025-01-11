import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# 生成一些示例数据
# 设置随机数生成器的种子（seed）,使用 42 作为种子，每次执行代码时，生成的随机数据都会一致。
# np.random.seed(42)
# data = np.random.rand(10, 2)  # 10个点的二维数据
# print(data)

# data = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
data = [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0], [5, 8], [5, 9], [6, 8], [6, 9]]

# 将数据 data 进行层次聚类，返回一个层次聚类的结果 Z。使用 ward 方法来计算簇之间的距离（或者说是相似性）。
Z = linkage(data, method='ward')  # "single"
print(type(Z), Z)

# 设置图形大小，宽10英寸，高7英寸
plt.figure(figsize=(10, 7))
# 将层次聚类的结果 Z 可视化成树状图
dendrogram(Z)
plt.show()
