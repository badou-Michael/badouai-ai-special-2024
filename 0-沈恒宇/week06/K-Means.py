"""
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :这个条件表示算法在达到指定的准确度（epsilon）时停止。
         在K均值聚类中，这个准确度通常表示簇中心之间的最小变化量。
         —-cv2.TERM_CRITERIA_MAX_ITER：这个条件表示算法在达到指定的最大迭代次数时停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    函数的返回值包括：
    retval：最终迭代时的紧凑性（即所有点到其聚类中心的距离之和）。
    bestLabels：每个数据点的聚类标签。
    centers：最终的聚类中心的位置，将包含 K 个二维点，这些点是聚类算法确定的聚类中心。
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像灰度颜色
img = cv2.imread('lenna.png',0)  # 0：以灰度模式读取 1：以彩色模式读取
print(img.shape)  # shape():灰度图像返回元组，包括图片的高度，宽度;彩色图像返回高度，宽度，通道数

# 获取图像高度、宽度
height, width = img.shape[:]

# 图像二维像素转换为一维
data = img.reshape((height*width,1))  # 表示将数组重新排列成一个 height*width 行、1 列的二维数组。
data = np.float32(data)  # 转换为float32位浮点数（数据类型），可以有效地节省内存，提高计算效率

# 停止条件(type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类，聚集成4类
compactness, labels, centers = cv2.kmeans(data,4,None,criteria,10,flags)

# 生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 默认字体为“SimHei”，即黑体

# 显示图像
title = [u'原始图像', u'聚类图像']  # 字符串可以使用前缀u来表示Unicode字符串
images = [img, dst]
for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])
    # plt.xticks([])：这个函数用于设置x轴的刻度线。传入一个空列表[]表示不显示任何刻度线。
    # plt.yticks([])：这个函数用于设置y轴的刻度线。同样传入一个空列表[]表示不显示任何刻度线。
plt.show()