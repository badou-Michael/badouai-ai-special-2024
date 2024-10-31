import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('lenna.png')
#print('image.shape:\n',image.shape)

#将image图像的高宽两个维度转换为一个维度
new_image=image.reshape(-1,3)
#Kmeans() 的data一定要是这个数据类型的
data=np.float32(new_image)
#print('new_image.shape:\n',new_image.shape)

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
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)


#compactness 紧密度，返回每个点到相应中心的距离的平方和
#labels 结果标记，每个成员被标记为分组的序号，如0，1，2，3，4...等
#centers 由聚类的中心组成的数据
#聚2类
compactness2,labels2,centers2=cv2.kmeans(data,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

#聚4类
compactness4,labels4,centers4=cv2.kmeans(data,4,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

#聚6类
compactness6,labels6,centers6=cv2.kmeans(data,6,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

#聚8类
compactness8,labels8,centers8=cv2.kmeans(data,8,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

#聚10类
compactness10,labels10,centers10=cv2.kmeans(data,10,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

#将图像还原回二维（指高宽）图像
#print(centers2)
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape(image.shape)

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape(image.shape)

centers6 = np.uint8(centers6)
res = centers6[labels6.flatten()]
dst6 = res.reshape(image.shape)

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape(image.shape)

centers10 = np.uint8(centers10)
res = centers10[labels10.flatten()]
dst10 = res.reshape(image.shape)

#将图像由BGR转为RGB
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2,cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4,cv2.COLOR_BGR2RGB)
dst6 = cv2.cvtColor(dst6,cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8,cv2.COLOR_BGR2RGB)
dst10 = cv2.cvtColor(dst10,cv2.COLOR_BGR2RGB)

#设置字体为SimHei以正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
title = ['原图','聚类图像 K=2','聚类图像 K=4','聚类图像 K=6','聚类图像 K=8','聚类图像 K=10']
image = [image,dst2,dst4,dst6,dst8,dst10,]

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(image[i])
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
