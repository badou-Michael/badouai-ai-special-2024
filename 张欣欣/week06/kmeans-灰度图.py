import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png",0)
print(img.shape)
# 图像宽高
rows,cols = img.shape[:]
date =img.reshape(rows*cols,1)
date = np.float32(date)

# 停止
criteria =(
    cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0
)

flag = cv2.KMEANS_PP_CENTERS
compaactness, labels, centers = cv2.kmeans(date,4,None,criteria,flag,10)
dst = labels.reshape((img.shape[0],img.shape[1]))
plt.rcParams['font.sans-serif']=['SimHei']

titles = [u'原始图像',u'聚类图像']
images = [img,dst]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
