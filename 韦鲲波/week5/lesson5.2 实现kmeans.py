import os
os.environ["OMP_NUM_THREADS"] = '1'  # 解决KMeans警告，临时添加环境变量，必须加在import KMeans之前
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans


def plt_out(px, py, color=None):
    plt.figure()  # 创建一个图形
    plt.scatter(px, py, c=color, marker='x')  # 用scatter绘制散点图
    plt.title('KMeans')  # 标题
    # plt.grid(True)  # 显示网格
    plt.show()  # 输出


if __name__ == '__main__':
    rng = np.random.default_rng(5)
    P = rng.random((60, 2)) * 100
    # print(P)

    # 需要将x和y分开
    x = P[:, 0]
    y = P[:, 1]

    K = 4
    result = KMeans(n_clusters=K)
    result = result.fit_predict(P)
    # print(result)
    # q = plt_out(x, y, result)

    '''
    函数签名
    cv2.kmeans(data, K, bestLabels, criteria, attempts, flags[, centers]) -> retval, bestLabels, centers
    参数说明
    data: 输入数据，必须是 np.float32 类型的二维 NumPy 数组。每一行代表一个样本，每一列代表一个特征。对于图像处理任务，通常是一个像素强度值的集合。
    K: 簇的数量（即你希望将数据分成多少个簇）。
    bestLabels: 输出参数，表示每个样本所属的簇标签。它应该是一个形状为 (n_samples,) 的整数数组，类型为 np.int32 或 np.int64。如果传入 None，OpenCV 会自动创建这个数组。
    criteria: 停止条件，定义了算法何时停止迭代。它是一个包含三个元素的元组 (type, max_iter, epsilon)：
    type: 停止条件的类型，可以是 cv2.TERM_CRITERIA_EPS（基于精度）、cv2.TERM_CRITERIA_MAX_ITER（基于最大迭代次数），或者它们的组合 cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER。
    max_iter: 最大迭代次数。
    epsilon: 精度阈值，当簇中心的变化小于这个值时，算法停止。
    attempts: 尝试的次数。cv2.kmeans 会在每次尝试中随机初始化质心，并选择具有最小紧凑度（所有样本到其最近簇中心的距离之和）的结果。
    flags: 初始质心的选择方法：
    cv2.KMEANS_PP_CENTERS: 使用 k-means++ 初始化方法（推荐）。
    cv2.KMEANS_RANDOM_CENTERS: 随机选择初始质心。
    centers: 输出参数，表示每个簇的质心。它是一个形状为 (K, n_features) 的浮点数数组。如果传入 None，OpenCV 会自动创建这个数组。
    返回值
    retval: 紧凑度（所有样本到其最近簇中心的距离之和），这是一个标量值。
    bestLabels: 每个样本所属的簇标签数组。
    centers: 每个簇的质心数组。
    '''

    '''
    img = cv2.imread('lenna.png', 0)

    height, width = img.shape
    print(height, width)
    # data = img.astype(np.float32)
    data = img.reshape((height * width, 1)).astype(np.float32)
    print(data)
    K = 4
    bestLabels = None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1,
    )
    attempts = 10
    flags = cv2.KMEANS_PP_CENTERS

    bestLabels = cv2.kmeans(data, K, bestLabels, criteria, attempts, flags)[1]
    print(bestLabels)
    result = bestLabels.reshape((height, width))
    print(result)

    plt.imshow(result, 'gray')
    plt.imshow(img, 'gray')
    plt.show()
    '''

    img = cv2.imread('lenna.png')

    height, width = img.shape[0], img.shape[1]
    print(height, width)
    # data = img.astype(np.float32)
    data = img.reshape((-1, 3)).astype(np.float32)
    print(data)
    K = 2
    bestLabels = None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1,
    )
    attempts = 10
    flags = cv2.KMEANS_PP_CENTERS

    l = []
    for i in range(6):
        q2 = cv2.kmeans(data, 2 ** (i + 1), bestLabels, criteria, attempts, flags)[1]
        q3 = cv2.kmeans(data, 2 ** (i+1), bestLabels, criteria, attempts, flags)[2].astype(np.uint8)
        tmp = q3[q2.flatten()]
        result = tmp.reshape((img.shape))
        l.append(result)

    output1 = np.hstack((l[0], l[1], l[2]))
    output2 = np.hstack((l[3], l[4], l[5]))

    output = np.vstack(( output1, output2))

    cv2.imshow('lenna.png', img)
    cv2.imshow('lenna.png2', output)
    cv2.waitKey(0)








