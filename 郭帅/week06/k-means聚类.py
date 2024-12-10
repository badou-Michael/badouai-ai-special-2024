#k-means聚类

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#生成50个1-100之间的随机数
data = np.random.randint(1,100,(50,2))
#设置k-means聚类数
kmeans = KMeans(n_clusters=3)
#fit 执行k-means;_predict并获取每个数据点的簇标签
x = kmeans.fit_predict(data)

# 输出每个数据点及其对应的簇标签
#for i, label in enumerate(x):
    # data[i] 是一个包含单个元素的数组，data[i][0] 取出这个元素
    #print(f"数据点: {data[i][0]}, 簇标签: {label}")

# 绘制散点图
plt.scatter(data[:, 0], data[:, 1], c=x, cmap='viridis', marker='o', s=100)
plt.title('KMeans 聚类结果分布')
plt.xlabel('数据点值：第一列')
plt.ylabel('数据点值：第二列')
# 显示聚类的中心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='簇中心')
# 添加图例，因为要显示label，如果没有设置label，可以删掉
plt.legend()
# 显示图像
plt.show()
