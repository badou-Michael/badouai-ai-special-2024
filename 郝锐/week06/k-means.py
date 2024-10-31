#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/10/31 15:09
# @Author: Gift
# @File  : K-means.py 
# @IDE   : PyCharm
from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt
# seed函数用于指定随机数生成器的种子，以保证每次运行代码时生成的随机数序列是一致的
np.random.seed(0)
# 生成随机数据,np.random.randn(100, 2)生成100行2列的随机数，np.array([2, 2])表示每个点的初始位置
data = np.vstack((np.random.randn(100, 2) * 0.7 + np.array([2, 2]),
                  np.random.randn(100, 2) * 0.5 + np.array([-2, -2]),
                  np.random.randn(100, 2) * 0.3 + np.array([2, -2])))
#print(data)
#print(data.shape)
# k-means算法
def k_means(data, k, max_iters=100):
    """
    K-means算法
    :param data: 输入数据,数组形状为data.shape=(n_samples, n_features)
    :param k: 聚类的簇数
    :param max_iters: 最大迭代次数，默认100
    :return:centroids（聚类中心）, labels（每个数据点所属的簇标签）
    """
    n_samples, n_features = data.shape #(300,2)
    # 随机初始化k个聚类中心,随机抽取三个样本
    centroids = data[np.random.choice(range(n_samples), k, replace=False)]
    for i in range(max_iters):
        # 计算每个样本点到聚类中心的距离
        distances = np.sqrt(((data[:,np.newaxis] - centroids)**2).sum(axis=2))
        # 找到距离最近的聚类中心
        labels = np.argmin(distances, axis=1)
        # 更新聚类中心
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        # 如果聚类中心不再发生变化，则停止迭代
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels
#设置聚类的簇数
k = 3
#调用k-means算法
centroids, labels = k_means(data, k)
#print(centroids)
#print(labels)
#绘制聚类结果
plt.subplot(1, 3, 1)
plt.title('manual k-means')
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o')
#plt.show()
#使用cv2。kmeans函数
#将数据转换为float32类型-cv2.kmeans要求输入的数据类型为np.float32
cv2_data = np.float32(data)
#print(cv2_data)
# 重塑数据形状为一维数组形式
#data = data.reshape((-1, data.shape[1]))
#print("+++++++++++++++++++++++")
#print(data)
# 设置终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
# 设置标签
# labels:一个一维数组，其长度等于数据点的数量，数组中的每个元素表示对应的数据点所属的簇标签，标签值从0到K - 1
# centers:一个二维数组，形状为(K, 2)（这里假设数据是二维的，维度与数据点的特征维度相同），表示聚类中心的坐标。
# cv2.KMEANS_RANDOM_CENTERS：指定初始聚类中心的生成方式为随机生成。
ret, cv2_labels, cv2_centers = cv2.kmeans(cv2_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
plt.subplot(1, 3, 2)
plt.title('cv2 k-means')
plt.scatter(data[:, 0], data[:, 1], c=cv2_labels)
plt.scatter(cv2_centers[:, 0], cv2_centers[:, 1], c='red', marker='s')
#plt.show()
#使用sklearn.cluster.KMeans函数
# 创建KMeans模型
kmeans = KMeans(n_clusters=k)
# 聚类数据
kmeans.fit(data)
# 获取聚类中心和标签
sk_centers = kmeans.cluster_centers_
sk_labels = kmeans.labels_
print(sk_centers)
print(sk_labels)
plt.subplot(1, 3, 3)
plt.title('sklearn k-means')
plt.scatter(data[:, 0], data[:, 1], c=sk_labels) # 标记每个数据点所属的簇
plt.scatter(sk_centers[:, 0], sk_centers[:, 1], c='red', marker='^') # 标记中心点
plt.show()
#print(data)
