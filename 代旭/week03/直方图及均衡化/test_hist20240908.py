import cv2
import matplotlib.pyplot as plt
import numpy as np

#直方图及均衡化
img = cv2.imread("lenna.png",1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
#print(hist)
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(img_gray)
plt.subplot(223)
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)
plt.xlim([0,256])#设置x坐标轴范围
plt.subplot(224)
plt.hist(img_gray.ravel(),256)
plt.show()
