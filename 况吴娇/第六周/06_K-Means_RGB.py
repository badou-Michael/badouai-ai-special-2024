# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像
img = cv2.imread('lenna.png')
print (img.shape)

#图像3维像素转换为2维
data = img.reshape((-1,3))
data = np.float32(data)
    # -1：这个值告诉 NumPy 自动计算这一维度的大小，以便保持总元素数量不变
    #  其形状通常是 (height, width, channels)，其中 channels 通常是 3（对于彩色图像）。因此，-1 会自动计算为 (height * width)，即图像中的总像素数。
    # 3：这个值表示新数组的每个元素（或称为行）将包含 3 个通道的数据，对应于图像的 RGB 值。
    #彩色图像在 OpenCV 中通常是一个三维数组。这个三维数组的三个维度分别代表图像的高度、宽度和颜色通道。对于标准的BGR彩色图像，颜色通道的顺序是蓝（Blue）、绿（Green）、红（Red）
    #data 数组的维度：当你使用 reshape((-1, 3)) 后，你将每个像素的三个颜色通道值（RGB）放在同一行中，形成一个二维数组。
     #这个二维数组的形状是 (pixels, 3)，其中 pixels 是图像中的总像素数（即 height * width）。


#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS  ##cv2.KMEANS_RANDOM_CENTERS表示初始质心是随机选择的。
###分别对图像数据进行不同类别数（2、4、8、16、64）的 K-Means 聚类。
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


'''
#图像转换回uint8二维类型
centers 是一个二维数组，包含了聚类中心。
labels 是一个一维数组，包含了每个像素点的聚类标签。
labels.flatten() 将 labels 数组展平成一维数组。
centers[labels.flatten()] 使用 labels 数组中的索引从 centers 数组中选择对应的聚类中心值，为每个像素点分配正确的颜色值。
res 是一个一维数组，包含了根据聚类标签选择的聚类中心值。
res.reshape(img.shape) 将 res 数组重新塑形为原始图像的二维形状。

这些行代码将聚类中心的数据类型从浮点数转换为整数，因为图像的颜色值需要是整数。然后根据聚类标签将聚类中心的颜色值赋给每个像素，并重新转换为与原图像相同的二维形状。
'''

#图像转换回uint8二维类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))
'''
图像文件和图像处理库（如 OpenCV）通常使用无符号8位整数（uint8）来存储每个像素值。这是因为图像的每个通道（红、绿、蓝）的像素值范围是0到255，正好适合8位整数。
res = centers2[labels2.flatten()] 这一步，这是必要的，因为它将每个像素点的聚类标签映射到对应的聚类中心值。没有这一步，你将无法将聚类结果转换回图像格式。下面是详细解释：

labels2 数组包含了每个像素点的聚类标签，这些标签指示每个像素点属于哪个聚类中心。
labels2.flatten() 将 labels2 数组展平成一维数组，以便可以作为索引使用。
centers2[labels2.flatten()] 使用这些索引从聚类中心数组 centers2 中选择对应的聚类中心值，为每个像素点分配正确的颜色值。
如果不执行这一步，你将无法将聚类结果转换为图像格式，因为 centers2 数组只是包含了聚类中心的值，而没有指示哪些值属于哪些像素点。通过使用 labels2 数组，你可以将每个像素点与其对应的聚类中心值关联起来。

最后，res.reshape(img.shape) 这一步是必要的，因为它将一维数组 res 重新塑形为原始图像的二维形状（即 (height, width)），以便可以作为图像显示和处理。
'''
'''
将聚类中心转换为整数：使用 np.uint8 确保聚类中心的值是整数，适合作为图像的像素值。
根据聚类标签重构图像：使用 centers[labels.flatten()] 将每个像素点的聚类标签映射到对应的聚类中心值，然后使用 reshape 方法将一维数组转换回原始图像的二维形状。'''

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


##plt.subplot创建子图，plt.imshow显示图像，plt.title设置标题，plt.xticks和plt.yticks隐藏坐标轴。最后，plt.show()显示所有图像。