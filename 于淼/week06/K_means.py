import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    retval:返回聚类的总误差（或更具体的说，是各个点到其对应中心的距离的平方和），用于评估聚类的好坏。
    bestLabels:一个数组，存储每个数据点所属的聚类标签。标签是从 0 到 (K-1) 的整数，表示每个数据点被分到哪个聚类。
    centers:一个数组，包含最终计算得到的每个聚类的质心（中心点）的坐标。质心是每个聚类的平均位置，代表了该聚类的特征。
    
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
        --KMEANS_PP_CENTERS: 追求聚类质量,更适合于大数据集。
        --KMEANS_RANDOM_CENTERS: 随机选择，计算速度快，质量不一定稳定。
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

img = cv2.imread("F:\DeepLearning\Code_test\lenna.png",0)
width,height = img.shape[:]

# 将二维像素转换为一维
data = img.reshape((width * height,1))
data2 = np.float32(data)

k =2
attempts = 10
# 停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1)

# 初始中心的选择
flags = cv2.KMEANS_RANDOM_CENTERS

# Kmeans聚类，聚集成4类
retval,bestLables,centers = cv2.kmeans(data2,k,None,criteria,attempts,flags,None)

# 生成最终图像
Kmeans_img = bestLables.reshape(img.shape[0],img.shape[1])

# 用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

# 显示图像
title = [u'原始图像',u'聚类图像']  # 这里使用的 u 前缀表示这些字符串是 Unicode 字符串，确保它们能够正确显示中文字符。
imgs = [img,Kmeans_img]

for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(imgs[i],'gray'),plt.title(title[i])
    plt.xticks([]),plt.yticks([])    # 用于隐藏 x 轴和 y 轴刻度

plt.show()