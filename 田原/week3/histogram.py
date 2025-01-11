import cv2
import numpy as np
from matplotlib import pyplot as plt

# 灰度图像直方图
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.hist(gray.ravel(), 256)
plt.show()

#彩色图像直方图
image = cv2.imread("lenna.png")
cv2.imshow("Original",image)
chans = cv2.split(image)
colors = ("b","g","r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])
plt.show()

