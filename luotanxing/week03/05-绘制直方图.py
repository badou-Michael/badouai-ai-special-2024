import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('../week02/lenna.png')

#方法一
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure()# 新建一个新的图形窗口
data = img_gray.ravel()#展平数组
plt.hist(data, 256)# 0-256的直方图


#方法二
hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])# hist 为256 行 一列
plt.figure()#新建一个图像
plt.bar(np.arange(256), hist.ravel(),  width=0.8)#直方图(x为0到256 ,y值 ,宽度)
plt.show()

#方法三
chans = cv2.split(img)
colors = ("b","g","r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.bar(np.arange(256), hist.ravel(),  width=0.8 , color=color)
plt.show()