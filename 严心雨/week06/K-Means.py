# coding: utf-8

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.float32类型的N维点集
    K表示聚类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引。预设的标签分类或者None
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS(使用kmeans++算法的中心初始化算法) ;和cv2.KMEANS_RANDOM_CENTERS（每次随机选择初始中心）
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

#以灰度化打开图片
image = cv2.imread('lenna.png',0)
height , weight = image.shape
#print(image.shape)

#将图片从2维降为1维
data = image.reshape( height * weight , 1 )
data = np.float32(data) #cv2.kmeans 函数必须要该类型
print(data.shape)

#迭代停止的模式选择
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

#K-Means聚类 聚集成4类
#compactness 紧密度，返回每个点到相应中心的距离的平方和
#labels 结果标记，每个成员被标记为分组的序号，如0，1，2，3，4...等
#centers 由聚类的中心组成的数据
compactness,labels,centers=cv2.kmeans(data,4,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#生成最终图像
#将图像还原为二维图像
dst = labels.reshape(height,weight)

#设置字体为SimHei以正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
image_name=['原图像','聚类后图像']
image=[image,dst]
for i in range(0,2):
    plt.subplot(1,2,i+1),plt.imshow(image[i],'gray')
    plt.title(image_name[i])
    plt.xticks([]),plt.yticks([])
plt.show()
