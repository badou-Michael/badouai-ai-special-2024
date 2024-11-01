# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像
img = cv2.imread('../../images/imgLena.png')
print (img.shape)

#图像二维像素转换为一维
data = img.reshape((-1,3))
data = np.float32(data)
print("data:\n", data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

"""
data:
输入数据，应该是一个二维的 NumPy 数组，每一行表示一个样本，每一列表示一个特征。
例如，如果你有 100 个样本，每个样本有 2 个特征，那么 data 的形状应该是 (100, 2)。

K:
要分成的簇的数量。在这个例子中，我们指定为 2，即我们希望将数据分成 2 个簇。

bestLabels:
初始标签（可选）。如果设置为 None，则算法会自动选择初始质心。
通常可以忽略这个参数，直接传入 None。

criteria:
停止条件，是一个包含三个值的元组 (type, max_iter, epsilon)。

type: 停止条件的类型，可以是 cv2.TERM_CRITERIA_EPS（精度）或 cv2.TERM_CRITERIA_MAX_ITER（最大迭代次数）。

max_iter: 最大迭代次数。

epsilon: 精度，当两次迭代之间的误差小于这个值时停止。
例如，criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 表示当误差小于 1.0 或达到最大迭代次数 10 时停止。

attempts:
尝试的次数。在每次尝试中，算法会随机选择不同的初始质心。最终结果将是所有尝试中最优的结果。
在这个例子中，我们设置为 10，即算法会进行 10 次尝试，然后返回最好的结果。

flags:
标志位，用于控制算法的行为。常用的标志包括：
cv2.KMEANS_RANDOM_CENTERS: 初始质心随机选择。
cv2.KMEANS_PP_CENTERS: 使用 k-means++ 方法选择初始质心。
在这个例子中，我们使用的是 cv2.KMEANS_RANDOM_CENTERS。


返回值：
compactness:
紧凑度，即所有点到其最近质心的距离的平方和。这个值越小，说明聚类效果越好。

labels2:
每个样本所属的簇的标签。它是一个一维数组，长度与输入数据的样本数量相同。
例如，如果 labels2[i] == 0，表示第 i 个样本属于第一个簇；如果 labels2[i] == 1，表示第 i 个样本属于第二个簇。

centers2:
每个簇的中心点坐标。它是一个二维数组，形状为 (K, feature_count)，其中 K 是簇的数量，feature_count 是特征的数量。
例如，如果 centers2[0] 是 [x1, y1]，表示第一个簇的中心点坐标为 (x1, y1)。
"""
#K-Means聚类 聚集成2类
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)

#K-Means聚类 聚集成4类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)

#K-Means聚类 聚集成8类
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)

#K-Means聚类 聚集成16类
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)

#K-Means聚类 聚集成64类
compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

print(labels2)
#图像转换回uint8二维类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))

#图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16',  u'聚类图像 K=64']  
images = [img, dst2, dst4, dst8, dst16, dst64]  
for i in range(6):  
   plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray'), 
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()
