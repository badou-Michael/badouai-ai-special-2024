import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


# 1.实现透视变化
# 透视变换矩阵：目标点=变换矩阵*原始点=》[X,Y,Z]=变换矩阵*[x,y,z],
# 原图是2维，z=1，所以Z=a31*x+a32*y+a33
# 另Z=1，所以目标点=目标点/Z=目标点/a31*x+a32*y+a33
# X=a11*x+a12*y+a13*1 / a31*x+a32*y+a33
# Y=a21*x+a22*y+a23*1 / a31*x+a32*y+a33
# 另a33=1上式变为
# X=a11*x+a12*y+a13 / a31*x+a32*y+1
# Y=a21*x+a22*y+a23 / a31*x+a32*y+1
# 共八个未知数，因此选取4组对应变换点

def perspective_transformation():
    # 读图
    img = cv2.imread('photo1.jpg')
    cv2.imshow("img_src", img)
    # 设置4组变换对应点(从老师代码里获取的)
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    # 获取变换矩阵
    matrix = cv2.getPerspectiveTransform(src, dst)
    # 使用变换矩阵变换(参数分别是，原图、变换矩阵、目标图像大小)
    result = cv2.warpPerspective(img, matrix, (337, 488))
    cv2.imshow("img_dst", result)
    cv2.waitKey(0)

# 2.实现kmeans
# 确定数据分为k组
# 任选k个点作为初始质心
# 分别计算剩余点到上述质心的距离，并将其划分到与其距离最近的质心组
# 计算k组之心的平均值作为新的质心
# 重复上述两步骤，直至质心组的数据不在变换为止


def k_means(k, attempts):
    # 读图
    img = cv2.imread("lenna.png")
    cv2.imshow("img_src", img)
    # 二维转一维
    data = img.reshape((-1, 3))
    # 提高精度
    data = np.float32(data)
    # 停止条件 (type,max_iter,epsilon)
    # criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
    #         其中，type有如下模式：
    #          —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
    #          —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
    #          —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 选择初始质心：flags表示，两种方法是cv2.KMEANS_PP_CENTERS(基于中心化算法选取中心点)和cv2.KMEANS_RANDOM_CENTERS(随机选取中心点)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # retval, bestLabels, centers=cv2.kmeans(data, K, bestLabels,criteria, attempts,flags)
    #   data 表示输入的待处理数据集合，应是 np.float32 类型，每个特征放在单独的一列中。
    #   K 表示要分出的簇的个数，即分类的数目，最常见的是 K=2，表示二分类。
    #   bestLabels 表示计算之后各个数据点的最终分类标签（索引）。实际调用时，参数bestLabels 的值设置为 None。
    #   criteria：算法迭代的终止条件。
    #   attempts：指定 attempts 的值，可以让算法使用不同的初始值进行多次（attempts 次）尝试
    #   flags：表示选择初始中心点的方法
    # 返回值:
    #  retval：距离值（也称密度值或紧密度），返回每个点到相应中心点距离的平方和。
    #  bestLabels：各个数据点的最终分类标签（索引）。
    #  centers：每个分类的中心点数据。
    etval, bestLabels, centers = cv2.kmeans(data, k, None, criteria, attempts, flags)
    # print("bestLabels:", bestLabels.flatten())
    # 生成的数据再转回图像
    centers = np.uint8(centers)
    # print(centers)
    # 根据标签索引和每个组的质心值生成一维数据
    res = centers[bestLabels.flatten()]
    # print(res)
    # 一维数据还原至原图
    result = res.reshape(img.shape)
    cv2.imshow("result", result)
    cv2.waitKey(0)


# perspective_transformation()
k_means(2, 10)
