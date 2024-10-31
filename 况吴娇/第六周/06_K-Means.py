'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
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
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''
'''
K-Means聚类算法通常在一维或多维数值数据上操作，而不是直接在二维图像矩阵上。通过将图像转换为一维数组，
你可以将每个像素视为数据集中的一个单独点，其中每个点的特征是它的灰度值。这样做有几个原因：

简化数据结构：一维数组比二维数组更容易处理，特别是在使用某些机器学习算法时。

算法要求：K-Means算法需要输入数据为一维数组的形式，其中每个元素代表一个数据点。

性能：在某些情况下，一维数组可以提高算法的性能，因为它们通常需要更少的内存，并且可以更有效地进行处理。

特征表示：在一维数组中，每个像素可以被视为一个特征，这使得算法可以基于像素的灰度值对像素进行聚类。

可视化：虽然原始图像是二维的，但聚类结果可以更容易地在一维数组中可视化，因为每个像素都有一个唯一的标签。
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度颜色
img=cv2.imread('lenna.png',0)
if img is None:
   print("图像读取失败，请检查文件路径。")
   exit()
print(img.shape)
#获取图像高度、宽度
rows, cols = img.shape[:]
print('rows:',rows,'cols:',cols)

#图像二维像素转换为一维
data = img.reshape((rows*cols,1))
data=np.float32(data)

#停止条件 (type,max_iter,epsilon)
criteria=(cv2.TERM_CRITERIA_MAX_ITER +cv2.TERM_CRITERIA_EPS ,10,1.0)
##在 cv2.kmeans 函数的 criteria 参数中，你应该使用按位或操作（| 或 +），而不是按位与操作（&）与。
#或操作（| 或 +）
#cv2.TERM_CRITERIA_EPS：这个标志意味着算法将在中心点之间的距离变化小于指定的 epsilon 值时停止。
# epsilon 是一个很小的正数，用来控制算法的精度。cv2.TERM_CRITERIA_MAX_ITER：这个标志意味着算法将在达到指定的最大迭代次数时停止。
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER，这意味着算法将在满足以下两个条件之一时停止：
# 中心点之间的距离变化小于 1.0（epsilon 值）。
# 迭代次数达到 10 次。

##设置标签
flags=cv2.KMEANS_RANDOM_CENTERS
##cv2.KMEANS_RANDOM_CENTERS：表示在每次尝试时随机选择初始中心点。
# 也就是说，每次运行 K-Means 算法时，都会随机选择不同的初始中心点，这有助于增加找到全局最优解的可能性，因为 K-Means 算法的结果可能会受到初始中心点选择的影

#K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
#生成最终图像
centers = np.uint8(centers)
labels = centers[labels.flatten()]

dst = labels.reshape((img.shape[0], img.shape[1]))

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei'] # # 设置中文字体为黑体，以便正确显示中文
                # # plt.rcParams 允许用户修改 matplotlib 的配置字典，这个字典包含了 matplotlib 的所有配置选项。通过修改这个字典，可以改变图形的默认样式，比如颜色、字体、线型等。
#
#                 #K-Means聚类 聚集成4类  4 表示要形成的聚类中心的数量。None 表示 bestLabels 参数被省略，这意味着 OpenCV 将自动分配一个数组来存储每个数据点的聚类结果。
#
#                 # OpenCV 的 cv2.kmeans 函数返回三个值：
#                 #
#                 # ret：这是一个表示聚类中心的紧凑度的值。在 K-Means 算法中，紧凑度通常定义为所有点到其最近聚类中心的距离平方和。这个值可以用来评估聚类的质量，数值越小表示聚类结果越紧凑。
#                 #
#                 # labels：这是一个数组，包含了每个数据点所属的聚类中心的索引。labels 的形状与输入数据 data 的第一个维度相同，即 (n_samples,)，其中 n_samples 是数据点的数量。
#                 #
#                 # centers：这是一个数组，包含了最终的聚类中心点。centers 的形状是 (K, n_features)，其中 K 是聚类中心的数量，n_features 是每个数据点的特征数量。
#                 #
#                 # 因此，当你调用 cv2.kmeans(data, 4, None, criteria, 10, flags) 时，你将得到以下三个输出：
#                 #
#                 # compactness：聚类的紧凑度。
#                 # labels：每个数据点的聚类标签。
#                 # centers：聚类中心点。

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst] ###images 也是一个列表，包含了两个图像的数据：原始图像 img 和聚类后的图像 dst
for i in range(2):
    plt.subplot(2,2,i+1), plt.imshow(images[i], 'gray') ##plt.subplot(2,2,i+1) 可放4个
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

'''
for i in range(2): 这个循环是因为我们有两个图像需要显示（原始图像和聚类图像）。range(2) 生成一个序列 [0, 1]，对应于 titles 和 images 列表中的两个元素。
plt.subplot(1,2,i+1) 创建子图。参数 1,2 表示一行两列的布局，i+1 表示子图的位置（第一张图是1，第二张图是2）。
plt.imshow(images[i], 'gray') 显示图像，'gray' 参数表示图像是灰度图。
plt.title(titles[i]) 设置子图的标题。
plt.xticks([]),plt.yticks([]) 移除坐标轴的刻度，使图像显示更清晰。
plt.show() 显示最终的图像窗口，包含所有子图。
'''

