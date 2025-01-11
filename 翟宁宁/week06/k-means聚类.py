'''
使用OpenCV的Kmeans接口实现图像聚类
构建kmeans参数数据 data,k,bestLabels=none,att

在OpenCV中，Kmeans()函数原型如下所示：
ret, labels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data ：聚类数据集，最好是np.float32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria：迭代终止条件， 格式为元组（type类型, max_iter最大迭代次数, epsilon精度）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，使用不同的初始聚类中心执行算法的次数
    flags表示初始中心（质心）的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据


    返回值：
    ret 表示聚类的惯性，这是一个衡量聚类质量的指标
    labels 是一个数组，包含了每个数据点的簇标签
    centers 是一个数组，包含了每个簇的中心点（质心）
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
#1. data
img = cv2.imread('../images/lenna.png',cv2.IMREAD_GRAYSCALE)
r,c = img.shape
#需要把二维图像像素转一维 ，目的为了计算
data = img.reshape(r*c,1)
data = np.float32(data)
#2.k
k=4
#3.bestLabels = none
#4.criteria 迭代终止条件，选择两者组合
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1)

#5.att 使用不同的初始聚类中心执行算法的次数
att = 20


#6.flag
flag = cv2.KMEANS_RANDOM_CENTERS

ret ,labels,centers = cv2.kmeans(data,k,bestLabels=None,criteria=criteria,attempts=att, flags=flag)
#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()
