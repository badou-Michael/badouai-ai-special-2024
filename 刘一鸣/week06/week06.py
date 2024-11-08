'''
作业20241102：
1.实现透视变化(相对于例子，进行了顺时针旋转)
2.实现kmeans。
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

#1.实现透视变化(相对于例子，进行了顺时针旋转)
img=cv2.imread('photo1.jpg')
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst=np.float32([[488, 0], [488, 337], [0, 0],[0,337]])

#生成透视变换矩阵
matrix=cv2.getPerspectiveTransform(src,dst)
#进行变换
res=cv2.warpPerspective(img,matrix,(448,337))
#cv2.imshow("src", img)
#cv2.imshow("result", res)
#cv2.waitKey(0)

#2.实现kmeans。
img2=cv2.imread('lenna.png',0)

rows,cols=img2.shape[:]
data=img2.reshape((rows*cols,1))
data=np.float32(data)

criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers=cv2.kmeans(data,8,None,criteria,10,flags)

res2=labels.reshape((rows,cols))
plt.imshow(res2, 'gray')
plt.show()

