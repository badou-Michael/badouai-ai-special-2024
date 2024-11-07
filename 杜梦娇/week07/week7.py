
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import random
########################### 层次聚类 ###############################
##使用scipy库实现层次聚类
Data =  [[1,2],[3,2],[4,4],[1,2],[1,3]]
#进行层次聚类
matrix = linkage(Data,"ward")
print(matrix)
#绘制图像
dendrogram(matrix)
plt.title('Hierarchical Clustering (scipy)')
plt.show()

# 根据聚类数量切割树状图，这里我们选择2个聚类
labels = fcluster(matrix, t=2, criterion='maxclust')
print(labels)

#使用sklearn库实现层次聚类(分2类)
clusters = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
clusters.fit(Data)

labels = clusters.labels_
Data = np.array(Data)
plt.scatter(Data[:, 0], Data[:, 1], c=labels)
plt.title('Hierarchical Clustering (sklearn)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
################################ RANSAC  #######################################
#RANSAC---最小二乘法
def least_squares_fit(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
    return coefficients

def calculate_residuals(points, coefficients):
    x = points[:, 0]
    y = points[:, 1]
    err_per_point = (y - (coefficients[0] * x + coefficients[1])) ** 2
    return err_per_point

def ransac(data, iterations, threshold, n):
    best_model = None
    best_inliers = None
    best_inlier_count = 0

    for _ in range(iterations):
        # 随机选择n个点
        rand_idxs = random.sample(range(len(data)), n)
        sample = data[rand_idxs, :]

        # 使用最小二乘法拟合直线模型
        model_params= least_squares_fit(sample)

        # 计算模型的拟合误差
        residuals = calculate_residuals(data, model_params)

        # 将误差小于阈值的点视为内点
        inliers = data[residuals < threshold]

        # 如果内点数量大于之前的最佳模型，则更新最佳模型
        if len(inliers) > best_inlier_count:
            best_inlier_count = len(inliers)
            best_inliers = inliers
            best_model = model_params

    return best_model, best_inliers

# 生成带有噪声的数据
np.random.seed(42)
x = np.arange(0, 300, 0.5)
y = 3 * x + 10 + np.random.normal(0,60, size=len(x))
# 添加一些异常值
outliers_index = np.random.choice(len(x), size=150, replace=False)
print(outliers_index)
outliers_value = np.random.normal(0, np.max(y)/2, 150)
print(outliers_value)
y[outliers_index] = outliers_value
# 合并数据
data = np.column_stack((x, y))
#拟合已知直线模型
model = least_squares_fit(data)
# 运行RANSAC算法
best_model, best_inliers = ransac(data, 100, 10, 5)

# 结果可视化
fig, ax = plt.subplots(figsize=(10, 6))
# 绘制原始数据
ax.scatter(x, y, label='Data', color='blue')
# 绘制RANSAC拟合的直线
if best_model is not None:
    m, b = best_model
    ransac_line = m * x + b
    ax.plot(x, ransac_line, label='RANSAC Line', color='red')
# 绘制最小二乘法拟合的直线
if model is not None:
    m, b = model
    ransac_line = m * x + b
    ax.plot(x, ransac_line, label='FIT Line', color='black')
#绘制期望模型
ax.plot(x, 3 * x + 2, label='EXACT Line', color='pink')
# 标记RANSAC内点
if best_inliers is not None:
    inlier_x, inlier_y = best_inliers[:, 0], best_inliers[:, 1]
    ax.scatter(inlier_x, inlier_y, label='RANSAC Inliers', color='green')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend(loc='best')
plt.show()
