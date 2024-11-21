import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


# 使用scipy实现层次聚类
def hierarchical_clustering_scipy():
    # 定义一个随机数种子，按固定顺序生成随机数
    np.random.seed(0)
    # 生成随机数据
    X = np.random.rand(10, 2)
    # 计算距离矩阵
    distance_matrix = pdist(X)
    # 进行层次聚类
    # linkage方法：
    # 'single': 最近邻
    # 'complete': 最远邻
    # 'average': 平均距离
    # 'ward': 最小方差
    Z = linkage(distance_matrix, method='ward')
    # 绘制树状图
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()


# 示例使用
def main():
   # 使用scipy
    hierarchical_clustering_scipy()



if __name__ == '__main__':
    main()
