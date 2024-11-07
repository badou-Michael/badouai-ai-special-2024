import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("lenna.png",0)
row,col=img.shape

data=img.reshape((row*col,1))
data = np.float32(data)

stop=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

flags = cv2.KMEANS_RANDOM_CENTERS

compactness, labels, centers= cv2.kmeans(data, 4, None, stop, 10, flags)

outcome=labels.reshape((row,col))
plt.subplot(1,2,1)
plt.title("original")
plt.imshow(img,cmap="gray")

plt.subplot(1,2,2)
plt.title("after")
plt.imshow(outcome,cmap="gray")

plt.show()
