from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

''' histogram '''

# 灰色图 的直方图 法一-柱状图
imgGrey = cv2.imread("lenna.png",0)
plt.figure()
hist = plt.hist(imgGrey.ravel(),256)
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.show()

# 灰色图 的直方图 法二-折线图
# imgGrey = cv2.imread("lenna.png",0)
# hist = cv2.calcHist([imgGrey],[0],None,[256],[0,256])
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.xlim([0,256])
# plt.plot(hist)
# plt.show()

# 彩色图 的直方图
img = cv2.imread("lenna.png",1)
chans = cv2.split(img)
colors = ["b","g","r"]
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0,256])

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist,color=color)
plt.show()
