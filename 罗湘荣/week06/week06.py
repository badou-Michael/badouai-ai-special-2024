import cv2
import numpy as np
import matplotlib.pyplot as plt

#实现透视变换
photo=cv2.imread("ho.jpg")

src=np.float32([[531, 42],[800, 50],[520, 215],[794,222]])
dst=np.float32([[0, 0], [840, 0], [0, 300], [840, 300]])

im=cv2.getPerspectiveTransform(src,dst)

result=cv2.warpPerspective(photo, im , (840,300))

cv2.imshow("Original", photo)
cv2.imshow("result", result)


#实现K-Means
photo1=cv2.imread("ho.jpg",0)

wide,high=photo1.shape[:]  #获取宽高

data=photo1.reshape(wide*high,1)
data=np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)

flags=cv2.KMEANS_RANDOM_CENTERS #标签

compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

dst = labels.reshape((photo1.shape[0], photo1.shape[1])) #生成图像

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签

titles = [u'原始图像', u'聚类图像']
img = [photo1, dst]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(img[i], 'gray')
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()
