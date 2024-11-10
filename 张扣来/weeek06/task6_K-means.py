#coding = utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('../../../request/task2/lenna.png',0)
print(img.shape)
# 获取图像高度和宽度
rows, cols = img.shape[:]
# 将 img 重塑为一维数组
data = img.reshape((rows*cols,1))
data = np.float32(data)
print(type(data))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
'''
cv2.KMEANS_RANDOM_CENTERS 是 OpenCV 中 cv2.kmeans 函数的一个参数选项，
用于指定选择初始聚类中心点的方法。
具体来说，cv2.KMEANS_RANDOM_CENTERS 表示随机选择初始聚类中心点
'''
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv2.kmeans(data,4,None,criteria,10,flags)
dst = labels.reshape((img.shape[0],img.shape[1]))
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
titles = [u'原始图像',u'聚类图像']
imgs = [img,dst]
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),
    plt.yticks([])
plt.show()
'''
`cv2.kmeans` 是 OpenCV 中实现 K-Means 聚类算法的函数。以下是其参数的详细说明：
1. **data**：输入的样本数据集，必须是按行来组织数据的，且需要是 `np.float32` 类型。每个特征应该放在一列。
2. **K**：分类的类别数，即你想要将数据聚成多少个簇。
3. **bestLabels**：每一个样本的标签，为一个 `Mat` 对象，每一行是一个样本的标签（属于第几类别）。
4. **criteria**：聚类迭代的停止条件。它应该是一个含有3个成员的元组，它们是（type，max_iter，epsilon）:
   - **type**：终止的类型，有如下三种选择：
     - `cv2.TERM_CRITERIA_EPS`：只有精确度 epsilon 满足时停止迭代。
     - `cv2.TERM_CRITERIA_MAX_ITER`：当迭代次数超过阈值时停止迭代。
     - `cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER`：上面的任何一个条件满足时停止迭代。
   - **max_iter**：最大迭代次数。
   - **epsilon**：精确度阈值。
5. **attempts**：使用不同的起始标记来执行算法的次数。算法会返回紧密度最好的标记。紧密度也会作为输出被返回。
6. **flags**：用来设置如何选择起始中心。通常我们有两个选择：
   - `cv2.KMEANS_RANDOM_CENTERS`：随机选取初始化中心点。
   - `cv2.KMEANS_PP_CENTERS`：使用某一种算法来确定初始聚类的点。
   - `cv2.KMEANS_USE_INITIAL_LABELS`：使用用户自定义的初始点。
输出参数：
1. **compactness**：紧密度返回每个点到相应中心的距离的平方和。
2. **labels**：标志数组，每个成员被标记为0，1等。
3. **centers**：有聚类的中心组成的数组。

这些参数共同控制了 K-Means 聚类算法的行为，包括数据的输入、聚类的数量、迭代的终止条件、初始中心点的选择等。
通过调整这些参数，可以对不同的数据集进行有效的聚类分析。

'''