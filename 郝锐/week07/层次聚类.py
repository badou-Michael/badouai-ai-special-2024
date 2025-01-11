#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/11/6 12:44
# @Author: Gift
# @File  : 层次聚类1.py 
# @IDE   : PyCharm
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
from skimage import io
import cv2
'''
scipy.cluster.hierarchy.linkage(y, method='single', metric='euclidean', optimal_ordering=False)
1. 可以是一个(n, p)形状的数组，其中n是数据点的数量，p是每个数据点的特征数量，代表了要进行聚类的原始数据；
也可以是一个(n - 1, 4)形状的数组，这通常是之前一次聚类结果的 linkage 矩阵，用于进一步的聚类分析。
2. 指定 linkage 方法，用于计算簇之间的距离。可选的值有 'single'（单链接）、
'complete'（全链接）、'average'（平均链接）、'weighted'（加权链接）、'centroid'（质心链接）、'median'（中位数链接）等。
默认为 'single'。'ward' 方差最小化算法
单链接：两个簇之间的距离定义为两个簇中任意两个数据点之间的最小距离。这种方法容易形成链状结构，对于非凸形状的数据分布可能效果较好，但对噪声和离群点比较敏感。
全链接：两个簇之间的距离定义为两个簇中任意两个数据点之间的最大距离。它倾向于将紧凑的簇合并，
对离群点相对不敏感，但可能会使聚类结果过于紧凑，导致一些本应分开的簇被合并在一起。
平均链接：两个簇之间的距离定义为两个簇中所有数据点对之间距离的平均值。它综合考虑了簇内的所有数据点，相对较为平衡，对于各种形状的数据分布都有一定的适应性。
加权链接：与平均链接类似，但在计算距离时，会根据簇的大小对距离进行加权。较大的簇在计算距离时会有更大的权重，这在某些情况下可以避免小簇被过早地合并到大簇中。
质心链接：两个簇之间的距离定义为它们的质心（均值向量）之间的距离。这种方法在处理具有不同密度的数据分布时可能会出现问题，因为质心可能会受到簇内数据分布的影响。
中位数链接：两个簇之间的距离定义为两个簇中所有数据点之间距离的中位数。它对数据中的噪声和离群点有一定的鲁棒性，并且在处理非凸形状的数据分布时表现较好。
Ward 链接法倾向于产生紧凑且相对平衡的聚类，使得每个簇内的数据点之间的相似度较高，簇与簇之间的差异较明显。

'''
'''
scipy.cluster.hierarchy.fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)
1.由 linkage 函数生成的层次聚类结果矩阵，形状为 (n - 1, 4)，其中 n 是原始数据点的数量。这个矩阵记录了聚类过程中每次合并的信息。 
2.用于确定簇的划分阈值。根据不同的 criterion 参数，t 的含义有所不同：
当 criterion='inconsistent' 时，t 是不一致性系数的阈值。不一致性系数用于衡量合并的簇之间的相似度与它们内部子簇的相似度的差异程度。
如果不一致性系数超过了 t，则停止合并，形成当前的簇划分。
当 criterion='distance' 时，t 是簇间距离的阈值。如果两个簇之间的距离小于等于 t，则将它们合并为一个簇。
当 criterion='maxclust' 时，t 是期望得到的最大簇数量。函数会根据层次聚类结果，将数据点划分为不超过 t 个簇。
3. criterion：确定簇划分的准则，可以是 'inconsistent'、'distance'、'maxclust' 或 'monocrit'。默认为 'inconsistent'。
4.depth:当 criterion='inconsistent' 时，用于计算不一致性系数的深度。它指定了在层次聚类树中向上回溯的层数，以计算不一致性系数。默认为 2。
5.R:不一致性系数矩阵。如果提供了 R，则使用它来计算不一致性系数，而不是重新计算。默认为 None。
6.monocrit:当 criterion='monocrit' 时，用于确定簇划分的单个准则。它是一个函数，接受两个参数（簇的索引和簇的大小），并返回一个值，用于确定簇的划分。默认为 None。
'''
np.random.seed(0)
X = np.random.rand(20, 2)
# X = cv2.imread('lenna.png')
# X = X.reshape((-1,3)) ##图像二维像素转换为一维
# X = np.float32(X)
# print(X)
#rows, cols, channels = X.shape
#X = X.reshape(rows * cols, channels)
#X = X.astype('float32') 数据转换太费内存，需要256G内存，放弃
#print(X.shape)
print(X)

Z = linkage(X, 'ward')
f = fcluster(Z,4,'distance')
fig = plt.figure(figsize=(8, 5))
#绘制层次聚类的树状图，输入为linkage后的数据
dn = dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
print(Z)
plt.show()
