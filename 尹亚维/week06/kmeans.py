# coding: utf-8

"""
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示输入的聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数（聚类中心数量）
    bestLabels表示输出的整数数组，用于存储每个样本所属的聚类标签（用于存储每个样本的聚类标签索引）。如果传入 None，函数会自动分配内存。
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS 和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 第二个参数枚举值
# cv2.IMREAD_COLOR 或 1：加载彩色图像。任何透明度都将被忽略，这是默认标志。
# cv2.IMREAD_GRAYSCALE 或 0：以灰度模式加载图像。
# cv2.IMREAD_UNCHANGED 或 -1：加载图像，包括 alpha 通道（如果存在）。
# cv2.IMREAD_ANYCOLOR 或 4：返回具有任意颜色格式的图像。
# cv2.IMREAD_ANYDEPTH 或 2：返回 16 位/32 位图像，当图像具有相应深度时。
# cv2.IMREAD_LOAD_GDAL 或 8：使用 GDAL 驱动程序加载图像。
# cv2.IMREAD_REDUCED_GRAYSCALE_2 或 16：以 1/2 原始尺寸的灰度模式加载图像。
# cv2.IMREAD_REDUCED_COLOR_2 或 17：以 1/2 原始尺寸的彩色模式加载图像。
# cv2.IMREAD_REDUCED_GRAYSCALE_4 或 32：以 1/4 原始尺寸的灰度模式加载图像。
# cv2.IMREAD_REDUCED_COLOR_4 或 33：以 1/4 原始尺寸的彩色模式加载图像。
# cv2.IMREAD_REDUCED_GRAYSCALE_8 或 64：以 1/8 原始尺寸的灰度模式加载图像。
# cv2.IMREAD_REDUCED_COLOR_8 或 65：以 1/8 原始尺寸的彩色模式加载图像

# 以灰度模式加载图像
img = cv2.imread('lenna.png', 0)
print(f"img.shape={img.shape}")

# 返回一个元组，表示图像的维度，获取图像高度、宽度
rows, cols = img.shape[:]

# 将图像img重塑为一个二维数组，其中第一维的大小为 rows * cols，第二维的大小为 1（通常用于灰度图像）
data = img.reshape((rows * cols), 1)
print(f"data_before={data}")
# 转换为float32数组
data = np.float32(data)
print(f"data_after={data}")

# —–cv2.TERM_CRITERIA_EPS: 精确度（误差）满足epsilon停止。
# —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
# —-cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
# 停止条件 (type, max_iter, epsilon) 迭代次数为10，精确度为1
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签，初始中心的选择：两种方法是cv2.KMEANS_PP_CENTERS和cv2.KMEANS_RANDOM_CENTERS
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类:第二个参数：K表示聚集成4类，第五个参数：attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签

# 紧凑度（Compactness）是 k-means 聚类算法的一个评价指标，表示所有数据点到它们所属聚类中心的距离平方和的平均值。
# 作用：紧凑度越小，说明聚类效果越好，因为数据点与其所属聚类中心的距离越近。
# 类型：浮点数。

# labels：每个数据点所属的聚类标签。
# 作用：用于标识每个数据点属于哪个聚类。
# 类型：一维整数数组，长度与输入数据点的数量相同。

# centers：每个聚类的中心点。
# 作用：表示每个聚类的代表点，通常用于可视化或进一步分析。
# 类型：二维数组，形状为 (K, n_features)，其中 K 是聚类的数量，n_features 是每个数据点的特征数量。
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
print(f"compactness={compactness}, labels={labels}, centers={centers}")

# 生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

print(f"dst={dst}")
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    # 创建一个1行2列的子图布局，并选择第i + 1个子图。
    # 参数：1：表示子图的行数。2：表示子图的列数。i + 1：表示当前选择的子图编号（从1开始计数）。
    # 在当前子图中显示images列表中的第i个图像，并使用灰度模式显示
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
    # 为当前子图设置标题
    plt.title(titles[i])
    # 隐藏坐标轴，移除当前子图的 x 轴和 y 轴刻度
    plt.xticks([]), plt.yticks([])
plt.show()
