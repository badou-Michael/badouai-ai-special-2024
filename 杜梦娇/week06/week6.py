
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np
import cv2
########################### 透视变换 ###############################
##透视变换
image = cv2.imread('photo1.jpg')

src_points = np.float32([[205, 152], [518, 284], [347, 732], [15, 604]])
dst_points = np.float32([[50, 50], [650, 50], [650, 850], [50, 850]])

#获取变换矩阵
warpMatrix = cv2.getPerspectiveTransform(src_points, dst_points)
#应用透视变换
warpedImage = cv2.warpPerspective(image, warpMatrix, (700, 900))

#绘制图像
cv2.imshow('warpedImage', warpedImage)
cv2.waitKey(0)


################################ k-means 聚类  #######################################
#k-means聚类---鸢尾花数据集进行聚类
# 导入数据
irisData = load_iris()
X = irisData.data
print(irisData.target_names)
# 定义K-means模型，设置聚类中心的数量
kmeans = KMeans(n_clusters=3, n_init='auto')
# 拟合模型
kmeans.fit(X)
# 预测每个样本的聚类标签
labels = kmeans.predict(X)
# 获取聚类中心
center = kmeans.cluster_centers_
# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(center[:, 0], center[:, 1], c='red')  # 聚类中心
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


#k-means聚类---图像分割
#导入图片数据、将图片转为RGB通道、获取图像大小信息并初始化变量
img = cv2.imread("lenna.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# 设置终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#灰度图像
img_gray_values = img_gray.reshape((-1, 1))
img_gray_values = np.float32(img_gray_values)
# 运行k-means 聚类(k=2)
ret1,label1,center1 = cv2.kmeans(img_gray_values,2,None,criteria,100,cv2.KMEANS_RANDOM_CENTERS)
# 运行k-means 聚类(k=4)
ret2,label2,center2 = cv2.kmeans(img_gray_values,4,None,criteria,100,cv2.KMEANS_RANDOM_CENTERS)
# 运行k-means 聚类(k=8)
ret3,label3,center3 = cv2.kmeans(img_gray_values,8,None,criteria,100,cv2.KMEANS_RANDOM_CENTERS)
# 运行k-means 聚类(k=16)
ret4,label4,center4 = cv2.kmeans(img_gray_values,16,None,criteria,100,cv2.KMEANS_RANDOM_CENTERS)
# 运行k-means 聚类(k=32)
ret5,label5,center5 = cv2.kmeans(img_gray_values,32,None,criteria,100,cv2.KMEANS_RANDOM_CENTERS)
# 将聚类中心转换为 uint8 类型
center1 = np.uint8(center1)
center2 = np.uint8(center2)
center3 = np.uint8(center3)
center4 = np.uint8(center4)
center5 = np.uint8(center5)
# 将聚类标签转换回图像格式
dst1 = center1[label1.flatten()]
dst1 = dst1.reshape(img_gray.shape)
dst2 = center2[label2.flatten()]
dst2 = dst2.reshape(img_gray.shape)
dst3 = center3[label3.flatten()]
dst3 = dst3.reshape(img_gray.shape)
dst4 = center4[label4.flatten()]
dst4 = dst4.reshape(img_gray.shape)
dst5 = center5[label5.flatten()]
dst5 = dst5.reshape(img_gray.shape)
# 显示分割后的图像
title_list = [u'Origin', u'K=2', u' K=4', u'K=8',  u'K=16',  u'K=32']
image_list = [img_gray, dst1, dst2, dst3, dst4, dst5]
for i in range(len(image_list)):
   plt.subplot(2,3,i+1), plt.imshow(image_list[i], 'gray'),
   plt.title(title_list[i])
   plt.xticks([]),plt.yticks([])
plt.show()


#彩色图像
# 将彩色三通道图像转换为浮点型并调整形状
img_values = img_rgb.reshape((-1, 3))
img_values = np.float32(img_values)

# 运行k-means 聚类(k=2)
_, lab1, cent1 = cv2.kmeans(img_values, 2, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
# 运行k-means 聚类(k=4)
_, lab2, cent2 = cv2.kmeans(img_values, 4, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
# 运行k-means 聚类(k=16)
_, lab3, cent3 = cv2.kmeans(img_values, 16, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
# 运行k-means 聚类(k=32)
_, lab4, cent4 = cv2.kmeans(img_values, 32, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
# 运行k-means 聚类(k=64)
_, lab5, cent5 = cv2.kmeans(img_values, 64, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

# 将聚类中心转换为 uint8 类型
centers1 = np.uint8(cent1)
centers2 = np.uint8(cent2)
centers3 = np.uint8(cent3)
centers4 = np.uint8(cent4)
centers5 = np.uint8(cent5)
# 将聚类标签转换回图像格式
segmented_image1 = centers1[lab1.flatten()]
segmented_image1 = segmented_image1.reshape(img_rgb.shape)
segmented_image2 = centers2[lab2.flatten()]
segmented_image2 = segmented_image2.reshape(img_rgb.shape)
segmented_image3 = centers3[lab3.flatten()]
segmented_image3 = segmented_image3.reshape(img_rgb.shape)
segmented_image4 = centers4[lab4.flatten()]
segmented_image4 = segmented_image4.reshape(img_rgb.shape)
segmented_image5 = centers5[lab5.flatten()]
segmented_image5 = segmented_image5.reshape(img_rgb.shape)
# 显示分割后的图像
title_list = [u'Origin', u'K=2', u' K=4', u'K=16',  u'K=32',  u'K=64']
image_list = [img_rgb, segmented_image1, segmented_image2, segmented_image3, segmented_image4, segmented_image5]
for i in range(len(image_list)):
   plt.subplot(2,3,i+1), plt.imshow(image_list[i]),
   plt.title(title_list[i])
   plt.xticks([]),plt.yticks([])
plt.show()
