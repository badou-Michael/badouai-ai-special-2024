import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:/Users/bq-twenty-one/Desktop/123.jpg")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure(1)
plt.imshow(img1),
plt.title(u'原始图像')


data = img.reshape((-1,3))
data = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
centers = np.uint8(centers)
res = centers[labels.flatten()]
dst = res.reshape((img.shape))
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

plt.figure(2)
plt.imshow(dst),
plt.title(u'聚类图像')
plt.show()
