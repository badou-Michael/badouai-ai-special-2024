import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 生成示例数据
np.random.seed(42)
X = np.random.rand(100, 2)  # 生成 100 个二维点

# 创建 K-means 模型,将数据分为 3 个簇
kmeans = KMeans(n_clusters=3, random_state=42)

# 拟合模型并预测聚类结果
y_kmeans = kmeans.fit_predict(X)

# 获取聚类中心点
centers = kmeans.cluster_centers_

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centers')
plt.title("K-means Clustering")
# 设置绘图的 x 轴和 y 轴标签
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
