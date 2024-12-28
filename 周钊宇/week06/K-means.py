import cv2
import numpy as np

img = cv2.imread("/Users/zhouzhaoyu/Desktop/ai/lenna.png")

#转换成一维数据

data = img.reshape((img.shape[0]*img.shape[1],3))
data = np.float32(data)
print(data.shape)

compactness, label, center = cv2.kmeans(data, 2, None, (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS,10,1.0),10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
dst = res.reshape((img.shape))


cv2.imshow("original image", img)
cv2.imshow("kmeans", dst)
cv2.waitKey(0)